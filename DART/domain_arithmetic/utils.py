import dataclasses
from functools import partial
import logging
from pathlib import Path
import shutil
from typing import Any

from flax import serialization
from flax.core import FrozenDict
from flax.traverse_util import flatten_dict
from flax.traverse_util import unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
import torch

# Safe import for flax.nnx
try:
    from flax import nnx

    HAS_NNX = True
except ImportError:
    HAS_NNX = False


# OpenPI imports
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


def check_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped existing output directory: {path}")
        else:
            raise FileExistsError(f"Output directory '{path}' already exists.\nUse --overwrite to wipe it and proceed.")
    else:
        path.mkdir(parents=True, exist_ok=True)


def load_policy_params(ckpt: Checkpoint) -> dict:
    """
    Loads the model using OpenPI loader and extracts the parameter dictionary.
    Supports both standard Flax Modules and flax.nnx Modules (e.g., Pi0).
    """
    print(f"Loading policy: {ckpt.config} from {ckpt.dir}")

    policy = _policy_config.create_trained_policy(_config.get_config(ckpt.config), ckpt.dir)

    # Access the internal model object
    model = policy
    if hasattr(policy, "_model"):
        model = policy._model  # noqa: SLF001
        print(f"  -> Found internal model type: {type(model).__name__}")

    # Detect and extract flax.nnx Module state
    is_nnx = hasattr(model, "iter_modules") or hasattr(model, "_graph_node_flatten")

    if is_nnx and HAS_NNX:
        print("  -> Detected flax.nnx Module. Using nnx.state()...")
        try:
            # Extract full graph state (Params + BatchStats, etc.)
            state = nnx.state(model)
            # Convert to pure dictionary
            return state.to_pure_dict()
        except Exception as e:
            print(f"  -> nnx extraction failed: {e}")

    # Fallback: Standard extraction methods (non-nnx)
    print("  -> Attempting standard extraction methods...")

    # Case: Model itself is a dictionary
    if isinstance(model, dict | FrozenDict):
        if "params" in model:
            return model["params"]
        return model

    # Case: Standard Flax attributes
    candidates = [
        ("model.params", lambda m: m.params),
        ("model.variables['params']", lambda m: m.variables["params"]),
        ("model.state.params", lambda m: m.state.params),
    ]

    for desc, accessor in candidates:
        try:
            data = accessor(model)
            data_dict = serialization.to_state_dict(data)
            if isinstance(data_dict, dict) and len(data_dict) > 0:
                print(f"  -> Successfully extracted params from: {desc}")
                return data_dict
        except Exception:
            continue

    # Last Resort: Manual component harvesting (specific to Pi0 architecture)
    print("  -> Attempting manual component harvesting...")
    manual_dict = {}
    key_components = ["PaliGemma", "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]

    for key in key_components:
        if hasattr(model, key):
            try:
                comp = getattr(model, key)
                # Recursively try nnx state or serialization
                comp_dict = nnx.state(comp).to_pure_dict() if is_nnx and HAS_NNX else serialization.to_state_dict(comp)

                if isinstance(comp_dict, dict) and comp_dict:
                    manual_dict[key] = comp_dict
            except Exception:
                pass

    if manual_dict:
        print(f"  -> Harvested components: {list(manual_dict.keys())}")
        return manual_dict

    # Extraction Failed
    print("\n" + "!" * 30)
    print("!!! EXTRACTION FAILED !!!")
    print(f"Model Type: {type(model)}")
    print(f"Available Attributes: {[d for d in dir(model) if not d.startswith('__')]}")
    print("!" * 30 + "\n")
    raise ValueError(f"Could not locate parameters in policy object: {ckpt.config}")


def save_params(params: Any, path: Path) -> None:
    """Saves the parameters dictionary as an Orbax checkpoint."""
    path = path.absolute()
    params_path = path / "params"
    print(f"Saving Orbax checkpoint to: {path}")

    save_payload = {"params": params}
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    if params_path.exists():
        logging.warning("Overwriting existing checkpoint...")

    checkpointer.save(params_path, save_payload)


def pytree_to_torch_cpu(params):
    flat = flatten_dict(params)
    flat_t = {}
    for k, v in flat.items():
        # v may be jax.Array; convert to np on CPU
        arr = np.array(v)  # materialize to host
        flat_t[k] = torch.from_numpy(arr.astype(float))  # CPU tensor shares memory if possible
    return flat_t


def torch_flat_to_jax_pytree(flat_torch, template_params):
    # template gives you desired dtype/structure
    flat_template = flatten_dict(template_params)
    out_flat = {}
    for k, t in flat_torch.items():
        # Convert to numpy then to jax
        out_flat[k] = jnp.asarray(t.detach().cpu().numpy())
        # optionally cast to template dtype
        out_flat[k] = out_flat[k].astype(flat_template[k].dtype)
    return unflatten_dict(out_flat)


def _is_layer_stacked_param(key: tuple, w) -> bool:
    """
    Heuristic for Flax/scan-packed layer parameters.

    Supports:
      - jax.numpy.ndarray
      - torch.Tensor
      - numpy.ndarray

    We treat a param as layer-stacked if:
      - key path suggests a scanned stack
      - tensor is at least 1D
      - leading dim is a plausible layer count
    """
    # ---- type / shape guards (backend-agnostic) ----
    if w is None:
        return False

    # ndim
    ndim = getattr(w, "ndim", None)
    if ndim is None or ndim == 0:
        return False

    # shape
    shape = getattr(w, "shape", None)
    if shape is None or len(shape) == 0:
        return False

    # ---- key-based heuristic ----
    key_strs = [str(k) for k in key]
    looks_scanned = "layers" in key_strs or "encoderblock" in key_strs
    if not looks_scanned:
        return False

    # ---- plausible layer count heuristic ----
    try:
        n0 = int(shape[0])
    except Exception:
        return False

    # Typical transformer depth range
    return 1 <= n0 <= 256


def randomized_svd(
    matrix: jnp.ndarray,
    rank: int,
    key: jax.Array,
    n_iter: int = 4,
    oversample: int = 10,
):
    """
    Randomized SVD (range finder + subspace iteration) returning (u, s, vh).

    matrix: (m, n), float32
    Returns:
      u: (m, rank)
      s: (rank,)
      vh: (rank, n)
    """
    assert matrix.ndim == 2, "Input matrix must be 2D."
    assert matrix.dtype == jnp.float32, "Assume float32 input."

    m, n = matrix.shape
    r = int(rank)
    k = int(min(r + oversample, m, n))
    if k <= 0:
        raise ValueError("Invalid rank/oversample for matrix shape.")

    omega = jax.random.normal(key, (n, k), dtype=jnp.float32)
    y = matrix @ omega

    q, _ = jnp.linalg.qr(y, mode="reduced")

    for _ in range(n_iter):
        z = matrix.T @ q  # (n, k)
        y = matrix @ z  # (m, k)
        q, _ = jnp.linalg.qr(y, mode="reduced")

    b = q.T @ matrix
    u_hat, s, vh = jnp.linalg.svd(b, full_matrices=False)

    u = q @ u_hat  # (m, k)
    return u[:, :r], s[:r], vh[:r, :]


@partial(jax.jit, static_argnames=["k"])
def top_k_prune(tensor: jnp.ndarray, k: float) -> jnp.ndarray:
    """
    Much faster implementation using hardware-optimized top_k.
    Calculates the keep count statically to avoid Tracer errors.
    """
    # 2. Compute size and num_keep in standard Python space
    # (tensor.size is static because JAX tensor shapes are static)
    size = tensor.size
    num_keep = max(1, int(k * size))

    flattened = tensor.ravel()
    flat_abs = jnp.abs(flattened)

    # Now num_keep is a standard integer, so top_k knows the exact output shape!
    top_values, _ = jax.lax.top_k(flat_abs, num_keep)

    threshold = top_values[-1]

    mask = jnp.abs(tensor) >= threshold

    return tensor * mask


def to_torch_like_2d(key: tuple, w, *, is_layer_stacked_fn=None):
    """
    Convert a Flax/JAX parameter leaf into a Torch-like 2D weight.
    Supports PaLI-Gemma / Flaxformer / ViT styles.
    """
    w = jnp.asarray(w)
    key_strs = [str(k) for k in key]

    def default_is_layer_stacked(k, arr):
        if arr.ndim < 3:
            return False
        looks_scanned = ("layers" in key_strs) or ("encoderblock" in key_strs)
        if not looks_scanned:
            return False
        n0 = int(arr.shape[0])
        return 1 <= n0 <= 256

    is_layer_stacked_fn = is_layer_stacked_fn or default_is_layer_stacked
    stacked = bool(is_layer_stacked_fn(key, w))

    def _has(sub: str) -> bool:
        return any(sub == s or sub in s for s in key_strs)

    def _one_to_out_in(x: jnp.ndarray) -> jnp.ndarray:
        # 0) 1D Parameters (Bias, Scale)
        if x.ndim == 1:
            return x

        # 1) Image Convolutional Patch Embedding
        if _has("embedding") and _has("kernel") and x.ndim == 4:
            out_ch = x.shape[3]
            # (H, W, In, Out) -> Transpose (Out, In, H, W) -> Flatten
            return jnp.transpose(x, (3, 2, 0, 1)).reshape(out_ch, -1)

        # 2) Positional or Input Embeddings
        if _has("embedding") or _has("pos_embedding"):
            # Pos Embedding (1, Seq, Dim) -> (Seq, Dim)
            if _has("pos_embedding") and x.ndim == 3 and x.shape[0] == 1:
                return x.squeeze(0)
            # Standard Embeddings (Vocab, Dim)
            if x.ndim == 2 and not _has("kernel"):
                return x

        # 3) Attention Q (LLM-style)
        if _has("q_einsum"):
            h_, d_model, hd = x.shape
            # (H,d_model,hd) -> (H*hd, d_model)
            return jnp.transpose(x, (1, 0, 2)).reshape(d_model, h_ * hd).T

        # 4) Attention KV (LLM-style)
        if _has("kv_einsum"):
            if x.ndim == 4:
                _two, kv_h, d_model, hd = x.shape
                return jnp.transpose(x, (2, 0, 1, 3)).reshape(d_model, 2 * kv_h * hd).T
            if x.ndim == 3:
                _two, d_model, hd = x.shape
                return jnp.transpose(x, (1, 0, 2)).reshape(d_model, 2 * 1 * hd).T

        # 5) Attention Output (LLM-style)
        if _has("attn_vec_einsum"):
            h_, hd, d_out = x.shape
            # (H, hd, d_out) -> (d_out, H*hd)
            return x.reshape(h_ * hd, d_out).T

        # 6) MLP Gated (LLM-style)
        if _has("gating_einsum"):
            _two, d_model, _d_ff = x.shape
            # (2, d_model, d_ff) -> (2*d_ff, d_model)
            # return jnp.transpose(x, (1, 0, 2)).reshape(d_model, 2 * d_ff).T
            # FIXME
            return x[0].T, x[1].T

        # 7) ViT Attention Kernels
        if _has("MultiHeadDotProductAttention"):
            # Output Projection
            if _has("out") and _has("kernel"):
                h_, hd, d_model = x.shape
                return x.reshape(h_ * hd, d_model).T

            # Input Q/K/V (not output)
            if not _has("out"):
                # Kernels
                if _has("kernel") and x.ndim == 3:
                    d_model, h_, hd = x.shape
                    # (d_model, H, hd) -> (H*hd, d_model)
                    return x.reshape(d_model, h_ * hd).T
                # Biases (Keys like: '...key', 'bias')
                if _has("bias") and x.ndim == 2:
                    # (H, head_dim) -> (H*head_dim,)
                    return x.ravel()

        # 8) Generic 2D Kernels (Dense, Linear)
        if x.ndim == 2 and (_has("kernel") or _has("w") or _has("linear") or _has("Dense")):
            # (In, Out) -> (Out, In)
            return x.T

        raise ValueError(f"Unhandled param for key={key} with shape={x.shape}")

    if stacked:
        return jax.vmap(_one_to_out_in, in_axes=0, out_axes=0)(w)
    return _one_to_out_in(w)


# -------- optional helpers if you want explicit K/V split like PyTorch --------
def kv_split_to_torch_like(key: tuple, w, *, is_layer_stacked_fn=None):
    """
    For kv_einsum leaves only: return (Wk, Wv) in torch-like (out,in) (or stacked (L,out,in)).
    """
    w = jnp.asarray(w)
    key_strs = [str(k) for k in key]
    if not any("kv_einsum" in s for s in key_strs):
        raise ValueError(f"kv_split_to_torch_like called on non-kv key={key}")

    def default_is_layer_stacked(k, arr):
        if arr.ndim < 4:
            return False
        looks_scanned = ("layers" in key_strs) or ("encoderblock" in key_strs)
        if not looks_scanned:
            return False
        n0 = int(arr.shape[0])
        return 1 <= n0 <= 256

    is_layer_stacked_fn = is_layer_stacked_fn or default_is_layer_stacked
    stacked = bool(is_layer_stacked_fn(key, w))

    def one(x):
        # x: (2,kvH,d_model,hd) or (2,d_model,hd)
        if x.ndim == 4:
            w_k = x[0]  # (kvH,d_model,hd)
            w_v = x[1]
            kv_h, d_model, hd = w_k.shape
            w_k2 = jnp.transpose(w_k, (1, 0, 2)).reshape(d_model, kv_h * hd).T
            w_v2 = jnp.transpose(w_v, (1, 0, 2)).reshape(d_model, kv_h * hd).T
            return w_k2, w_v2
        if x.ndim == 3:
            w_k = x[0]  # (d_model,hd)
            w_v = x[1]
            d_model, hd = w_k.shape
            return w_k.T, w_v.T
        raise ValueError(f"Unexpected kv shape {x.shape} for key={key}")

    if stacked:
        w_k, w_v = jax.vmap(one, in_axes=0, out_axes=0)(w)
        return w_k, w_v
    return one(w)


def restore_from_torch_like(key: tuple, w_merged: jnp.ndarray, original_shape: tuple):
    """
    Inverse of to_torch_like_2d.
    Restores the merged parameter (Out, In) back to original Flax shape.
    NOTE: Designed for single-layer restoration. If stacked, vmap this function.
    """
    key_strs = [str(k) for k in key]

    def _has(sub: str) -> bool:
        return any(sub == s or sub in s for s in key_strs)

    # 0) 1D Parameters
    if len(original_shape) == 1:
        return w_merged

    # 1) Image Convolutional Patch Embedding
    if _has("embedding") and _has("kernel") and len(original_shape) == 4:
        h_, w_, in_, out_ = original_shape
        # (Out, In*H*W) -> (Out, In, H, W)
        w_reshaped = w_merged.reshape(out_, in_, h_, w_)
        # Transpose -> (H, W, In, Out)
        return jnp.transpose(w_reshaped, (2, 3, 1, 0))

    # 2) Positional or Input Embeddings
    if _has("embedding") or _has("pos_embedding"):
        if _has("pos_embedding") and len(original_shape) == 3 and original_shape[0] == 1:
            # Merged: (Seq, Dim) -> Original: (1, Seq, Dim)
            return jnp.expand_dims(w_merged, axis=0)
        if len(original_shape) == 2 and not _has("kernel"):
            return w_merged

    # 3) Attention Q (LLM-style)
    if _has("q_einsum"):
        h_, d_model, hd = original_shape
        # (H*hd, d_model).T -> (d_model, H*hd)
        # Reshape -> (d_model, H, hd)
        # Transpose -> (H, d_model, hd)
        return jnp.transpose(w_merged.T.reshape(d_model, h_, hd), (1, 0, 2))

    # 5) Attention Output (LLM-style)
    if _has("attn_vec_einsum"):
        h_, hd, d_out = original_shape
        # (d_out, H*hd).T -> (H*hd, d_out)
        # Reshape -> (H, hd, d_out)
        return w_merged.T.reshape(h_, hd, d_out)

    # 6) MLP Gated
    if _has("gating_einsum"):
        _two, d_model, d_ff = original_shape
        # (2*d_ff, d_model).T -> (d_model, 2*d_ff)
        # Reshape -> (d_model, 2, d_ff)
        # Transpose -> (2, d_model, d_ff)
        return jnp.transpose(w_merged.T.reshape(d_model, 2, d_ff), (1, 0, 2))

    # 7) ViT Attention
    if _has("MultiHeadDotProductAttention"):
        # Output
        if _has("out") and _has("kernel"):
            h_, hd, d_model = original_shape
            return w_merged.T.reshape(h_, hd, d_model)

        # Q/K/V Inputs
        if not _has("out"):
            # Kernels
            if len(original_shape) == 3 and _has("kernel"):
                d_model, h_, hd = original_shape
                return w_merged.T.reshape(d_model, h_, hd)

            # Biases
            if len(original_shape) == 2 and _has("bias"):
                h_, hd = original_shape
                # Merged: (H*hd,) -> Original: (H, hd)
                return w_merged.reshape(h_, hd)

    # 8) Generic 2D (Dense, Linear)
    if len(original_shape) == 2 and (_has("kernel") or _has("w") or _has("linear") or _has("Dense")):
        return w_merged.T

    # Fallback for simple shapes
    if w_merged.shape == original_shape:
        return w_merged

    raise ValueError(f"Could not restore shape for key={key}, merged={w_merged.shape}, orig={original_shape}")


def restore_kv_einsum(merged_k, merged_v, original_shape):
    """
    Restores KV parameters split by kv_split_to_torch_like.
    """
    # Case A: (2, kvH, d_model, hd)
    if len(original_shape) == 4:
        _two, kv_h, d_model, hd = original_shape
        k_restored = jnp.transpose(merged_k.T.reshape(d_model, kv_h, hd), (1, 0, 2))
        v_restored = jnp.transpose(merged_v.T.reshape(d_model, kv_h, hd), (1, 0, 2))
        return jnp.stack([k_restored, v_restored], axis=0)

    # Case B: (2, d_model, hd) - Implicit kvH=1
    if len(original_shape) == 3:
        _two, d_model, hd = original_shape
        k_restored = merged_k.T
        v_restored = merged_v.T
        return jnp.stack([k_restored, v_restored], axis=0)

    raise ValueError(f"Unknown kv_einsum shape: {original_shape}")
