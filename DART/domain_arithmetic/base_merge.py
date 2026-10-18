import concurrent.futures
import dataclasses
import datetime
import functools
import gc
import json
import logging
from pathlib import Path
import shutil
from typing import Any

from flax.traverse_util import flatten_dict
from flax.traverse_util import unflatten_dict
import jax
from tqdm import tqdm

from openpi.shared import download
from openpi.training import config as _config

# Assumed imports from your local .utils
from .utils import Checkpoint
from .utils import _is_layer_stacked_param
from .utils import check_output_dir
from .utils import kv_split_to_torch_like
from .utils import load_policy_params
from .utils import restore_from_torch_like
from .utils import restore_kv_einsum
from .utils import save_params
from .utils import to_torch_like_2d

# Ensure CPU usage to avoid sharding errors during loading
NUM_WORKERS = 1  # adjust as needed
# For pi0.5, NUM_WORKERS * 12GB of RAM is needed

# jax.config.update("jax_platform_name", "cpu")


@dataclasses.dataclass
class BaseMergeConfig:
    """
    Base configuration meant to be inherited.
    Defines input/output paths required by the BaseMerging harness.
    """

    base_policy: Checkpoint
    policy_src: Checkpoint
    policy_tgt: Checkpoint

    seed: int = 42
    output_dir: str = "./merged_output"
    overwrite: bool = True

    # Scope Settings: "all", "vision", "no_action"
    merge_scope: str = "all"


class BaseMerging:
    """
    Base class handling the lifecycle of loading, iterating, JIT-compiling,
    and saving merged parameters.
    """

    def __init__(self, cfg: BaseMergeConfig):
        self.cfg = cfg

    def core_arithmetic_fn(self, w_base, w_src, w_tgt, rng_key, **kwargs):
        """
        Abstract method for the mathematical merge logic.
        Must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement core_arithmetic_fn")

    def _get_jit_static_args(self) -> dict[str, Any]:
        """
        Extracts configuration fields to be bound as static arguments for JAX JIT.
        By default, it takes all fields from the config that are NOT Checkpoints or paths.
        """
        static_args = {}
        for field in dataclasses.fields(self.cfg):
            value = getattr(self.cfg, field.name)
            # Exclude non-math config options (paths, objects) to prevent JIT re-compilation issues
            if field.name not in [
                "base_policy",
                "policy_src",
                "policy_tgt",
                "seed",
                "output_dir",
                "overwrite",
                "merge_scope",
            ]:
                static_args[field.name] = value
        return static_args

    def should_merge(self, key: tuple[str, ...]) -> bool:
        """
        Determines if a parameter should be merged based on the config scope.
        Returns True if it should be merged, False if it should be a copy of Policy A.
        """
        scope = self.cfg.merge_scope

        if scope == "all":
            return True

        # 1. Vision Scope: Only merge if 'img' is in the second position
        if scope == "vision":
            # Key structure ex: ('PaliGemma', 'img', ...)
            return bool(len(key) > 1 and key[1] == "img")

        # 2. No Action Scope: Exclude action experts
        if scope == "no_action":
            # Exclusion A: Root level action keys
            if len(key) > 0 and key[0] in ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]:
                return False

            # Exclusion B: LLM parameters ending with "_1" in the 4th position (index 3)
            # Key structure ex: ('PaliGemma', 'llm', 'layers', 'mlp_1', ...)
            if len(key) > 3 and key[0] == "PaliGemma" and key[1] == "llm":
                third_pos = key[2]
                if third_pos.endswith("_1"):
                    return False
                fourth_pos = key[3]
                if fourth_pos.endswith("_1"):
                    return False
                if len(key) > 4:
                    fifth_pos = key[4]
                    if fifth_pos.endswith("_1"):
                        return False

            # If not excluded, merge it
            return True
        if scope == "llm":
            # Exclusion A: Root level action keys
            if len(key) > 0 and key[0] in ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]:
                return False

            # Key structure ex: ('PaliGemma', 'img', ...)
            if len(key) > 1 and key[1] == "img":
                return False

            # Exclusion B: LLM parameters ending with "_1" in the 4th position (index 3)
            # Key structure ex: ('PaliGemma', 'llm', 'layers', 'mlp_1', ...)
            if len(key) > 3 and key[0] == "PaliGemma" and key[1] == "llm":
                third_pos = key[2]
                if third_pos.endswith("_1"):
                    return False
                fourth_pos = key[3]
                if fourth_pos.endswith("_1"):
                    return False
                if len(key) > 4:
                    fifth_pos = key[4]
                    if fifth_pos.endswith("_1"):
                        return False

            # If not excluded, merge it
            return True

        if scope == "llm_attn":
            return bool(len(key) == 6 and key[1] == "llm" and key[3] == "attn" and not key[4].endswith("_1"))

        if scope == "vision_llm_attn":
            # Key structure ex: ('PaliGemma', 'img', ...)
            if len(key) > 1 and key[1] == "img":
                return True

            return bool(len(key) == 6 and key[1] == "llm" and key[3] == "attn" and not key[4].endswith("_1"))

        if scope == "llm_mlp":
            return bool(len(key) == 5 and key[1] == "llm" and key[3] == "mlp" and not key[4].endswith("_1"))

        if scope == "vision_llm_mlp":
            # Key structure ex: ('PaliGemma', 'img', ...)
            if len(key) > 1 and key[1] == "img":
                return True

            return bool(len(key) == 5 and key[1] == "llm" and key[3] == "mlp" and not key[4].endswith("_1"))

        # Default fallback
        return True

    def merge_params(self):
        logging.info(f"Configuration: {self.cfg}")

        # 0. Prepare RNG
        master_key = jax.random.PRNGKey(self.cfg.seed)

        # 1. Load Checkpoints
        print("Loading checkpoints in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            future_base = executor.submit(load_policy_params, self.cfg.base_policy)
            future_src = executor.submit(load_policy_params, self.cfg.policy_src)
            future_tgt = executor.submit(load_policy_params, self.cfg.policy_tgt)

            params_base = future_base.result()
            params_src = future_src.result()
            params_tgt = future_tgt.result()

        # 2. Flatten Dictionaries
        print("Flattening dictionaries...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            future_base = executor.submit(flatten_dict, params_base)
            future_src = executor.submit(flatten_dict, params_src)
            future_tgt = executor.submit(flatten_dict, params_tgt)

            flat_base = future_base.result()
            flat_src = future_src.result()
            flat_tgt = future_tgt.result()

        keys_base = set(flat_base.keys())
        common_keys = list(keys_base.intersection(flat_src.keys(), flat_tgt.keys()))
        common_keys.sort()

        print(f"Total keys in base: {len(keys_base)}")
        print(f"Common keys to merge: {len(common_keys)}")

        # 3. Prepare JIT Compilation
        # We bind the static config arguments to the core arithmetic function here.
        static_args = self._get_jit_static_args()

        # Create a partial function with the static config args
        arithmetic_partial = functools.partial(self.core_arithmetic_fn, **static_args)

        # JIT compile: One for standard params, one for stacked params (vmapped)
        jit_single = jax.jit(arithmetic_partial)
        jit_stacked = jax.jit(jax.vmap(arithmetic_partial, in_axes=(0, 0, 0, 0)))

        # 4. Merge Loop
        merged_flat = {}
        print("Merging parameters...")
        for i, key in enumerate(tqdm(common_keys)):
            if (i + 1) % 10 == 0:  # tune this
                jax.clear_caches()
                gc.collect()

            w_base = flat_base[key]
            w_src = flat_src[key]
            w_tgt = flat_tgt[key]

            # --- FILTERING LOGIC ---
            if not self.should_merge(key):
                # If outside scope, copy Policy A as requested
                merged_flat[key] = w_tgt
                continue
            # -----------------------

            step_key = jax.random.fold_in(master_key, i)

            # Shape safety check
            if (w_base.shape != w_src.shape) or (w_base.shape != w_tgt.shape):
                merged_flat[key] = w_tgt
                continue

            # Reshape layers into (L, Out_dim, In_dim) or (Out_dim, In_dim) if needed
            key_strs = [str(k) for k in key]
            try:
                # --- CASE 1: KV Einsum (Split -> Merge -> Restore) ---
                if any("kv_einsum" in s for s in key_strs):
                    # Split into K and V (standard or stacked)
                    w_base_k, w_base_v = kv_split_to_torch_like(
                        key, w_base, is_layer_stacked_fn=_is_layer_stacked_param
                    )
                    w_src_k, w_src_v = kv_split_to_torch_like(key, w_src, is_layer_stacked_fn=_is_layer_stacked_param)
                    w_tgt_k, w_tgt_v = kv_split_to_torch_like(key, w_tgt, is_layer_stacked_fn=_is_layer_stacked_param)

                    # Merge
                    if _is_layer_stacked_param(key, w_base):
                        num_layers = w_base_k.shape[0]
                        layer_keys = jax.random.split(step_key, num_layers)

                        merged_val_k = jit_stacked(w_base_k, w_src_k, w_tgt_k, layer_keys)
                        merged_val_v = jit_stacked(w_base_v, w_src_v, w_tgt_v, layer_keys)

                        # Restore (vmap)
                        # Use lambda to avoid vmap argument collision
                        restore_fn = jax.vmap(
                            lambda k_val, v_val, w=w_base: restore_kv_einsum(k_val, v_val, w.shape[1:])
                        )
                        merged_val = restore_fn(merged_val_k, merged_val_v)
                    else:
                        merged_val_k = jit_single(w_base_k, w_src_k, w_tgt_k, step_key)
                        merged_val_v = jit_single(w_base_v, w_src_v, w_tgt_v, step_key)

                        # Restore (single)
                        merged_val = restore_kv_einsum(merged_val_k, merged_val_v, w_base.shape)

                    merged_flat[key] = merged_val
                elif any("gating_einsum" in str(s) for s in key):
                    # Split into K and V -> down linear and gate linear
                    w_base_k, w_base_v = to_torch_like_2d(key, w_base, is_layer_stacked_fn=_is_layer_stacked_param)
                    w_src_k, w_src_v = to_torch_like_2d(key, w_src, is_layer_stacked_fn=_is_layer_stacked_param)
                    w_tgt_k, w_tgt_v = to_torch_like_2d(key, w_tgt, is_layer_stacked_fn=_is_layer_stacked_param)

                    if _is_layer_stacked_param(key, w_base):
                        num_layers = w_base_k.shape[0]
                        layer_keys = jax.random.split(step_key, num_layers)

                        # Merge K and V separately (Stacked)
                        # Norms will be vectors of shape (num_layers,)
                        merged_val_k = jit_stacked(w_base_k, w_src_k, w_tgt_k, layer_keys)
                        merged_val_v = jit_stacked(w_base_v, w_src_v, w_tgt_v, layer_keys)

                        # Restore logic
                        restore_fn = jax.vmap(
                            lambda k_val, v_val, w=w_base: restore_kv_einsum(k_val, v_val, w.shape[1:])
                        )
                        merged_val = restore_fn(merged_val_k, merged_val_v)
                    else:
                        raise ValueError()
                    merged_flat[key] = merged_val
                # --- CASE 2: Standard Parameters (Convert -> Merge -> Restore) ---
                else:
                    w_base_2d = to_torch_like_2d(key, w_base, is_layer_stacked_fn=_is_layer_stacked_param)
                    w_src_2d = to_torch_like_2d(key, w_src, is_layer_stacked_fn=_is_layer_stacked_param)
                    w_tgt_2d = to_torch_like_2d(key, w_tgt, is_layer_stacked_fn=_is_layer_stacked_param)

                    if _is_layer_stacked_param(key, w_base):
                        num_layers = w_base_2d.shape[0]
                        layer_keys = jax.random.split(step_key, num_layers)

                        merged_2d = jit_stacked(w_base_2d, w_src_2d, w_tgt_2d, layer_keys)

                        # Restore (vmap)
                        restore_fn = jax.vmap(lambda m, k=key, w=w_base: restore_from_torch_like(k, m, w.shape[1:]))
                        merged_val = restore_fn(merged_2d)
                    else:
                        merged_2d = jit_single(w_base_2d, w_src_2d, w_tgt_2d, step_key)

                        # Restore (single)
                        merged_val = restore_from_torch_like(key, merged_2d, w_base.shape)

                    merged_flat[key] = merged_val

            except Exception as e:
                logging.error(f"Error processing key {key}: {e}")
                merged_flat[key] = w_base

        # 5. Handle Missing Keys (Copy from Base)
        missing_keys = keys_base - set(common_keys)
        if missing_keys:
            print(f"Copying {len(missing_keys)} missing keys from base...")
            for key in missing_keys:
                merged_flat[key] = flat_base[key]

        print("Unflattening...")
        return unflatten_dict(merged_flat)

    def save_results(self, merged_params):
        # Create Directory
        check_output_dir(Path(self.cfg.output_dir), overwrite=self.cfg.overwrite)
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)

        # Save Weights
        save_params(merged_params, Path(self.cfg.output_dir))

        # Copy norm stats from Policy A
        self._copy_norm_stats()

        # Save Metadata
        self._save_metadata()
        print(f"Merge Complete! Saved to: {self.cfg.output_dir}")

    def _save_metadata(self):
        # Generic metadata saver that serializes the config
        merge_info = {
            "timestamp": datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d %H:%M:%S"),
            "config": dataclasses.asdict(self.cfg),
            "output_dir": str(Path(self.cfg.output_dir).absolute()),
        }

        # Convert non-serializable objects (like Path/Checkpoint) to strings
        def json_default(obj):
            if isinstance(obj, Path | Checkpoint):
                return str(obj)
            raise TypeError

        json_path = Path(self.cfg.output_dir) / "merge_config.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(merge_info, f, indent=4, ensure_ascii=False, default=json_default)

    def _copy_norm_stats(self):
        """Copy norm stats from Policy A to the output directory."""
        # Download/resolve the policy A directory (handles GCS paths)
        policy_tgt_dir = Path(download.maybe_download(self.cfg.policy_tgt.dir))

        # Get the config to retrieve asset_id
        try:
            policy_a_config = _config.get_config(self.cfg.policy_tgt.config)
            asset_id = policy_a_config.data.assets.asset_id

            # Look for norm stats under assets/<asset_id>/
            assets_src = policy_tgt_dir / "assets" / asset_id

            if assets_src.exists() and assets_src.is_dir():
                assets_dst = Path(self.cfg.output_dir) / "assets" / asset_id
                print(f"Copying norm stats from {assets_src} to {assets_dst}...")
                shutil.copytree(assets_src, assets_dst, dirs_exist_ok=True)
            else:
                logging.warning(f"No assets directory found at: {assets_src}")
        except Exception as e:
            logging.warning(f"Could not copy norm stats: {e}")

    def run(self):
        merged_params = self.merge_params()
        self.save_results(merged_params)
