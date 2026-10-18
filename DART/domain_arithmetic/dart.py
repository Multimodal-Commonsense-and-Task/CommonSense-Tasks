import dataclasses
import logging

import jax
import jax.numpy as jnp
import tyro

from .base_merge import BaseMergeConfig
from .base_merge import BaseMerging


def _greedy_subspace_align_merge(
    tau_src: jnp.ndarray,
    tau_tgt: jnp.ndarray,
    rng_key: jnp.ndarray,
    scaling_coef: float,
) -> jnp.ndarray:
    """
    Performs Greedy Subspace selection on task vectors.
    Selects original columns of U_c that align with B without rotating/mixing them.
    """
    dtype_out = tau_tgt.dtype
    shape_out = tau_tgt.shape

    if tau_tgt.ndim <= 1:
        return scaling_coef * (tau_tgt - tau_src)

    tau_src_2d = tau_src.astype(jnp.float32)
    tau_tgt_2d = tau_tgt.astype(jnp.float32)

    rows, cols = tau_tgt_2d.shape
    min_dim = int(min(rows, cols))
    if min_dim <= 0:
        return jnp.zeros_like(tau_tgt)

    # Helper to compute projection: U U^T X
    def get_proj(tensor, u_basis):
        # We assume u_basis columns are either normalized or zeroed out
        return u_basis @ (u_basis.T @ tensor)

    def get_sar(tensor, basis, s, total_dim):
        s_sq = jnp.square(s)
        energy = jnp.cumsum(s_sq) / (jnp.sum(s_sq) + 1e-6)  # Avoid div by zero
        r_cutoff = jnp.searchsorted(energy, 0.9975)  # keep 95% energy to remove noise
        indices = jnp.arange(total_dim)
        mask = (indices <= r_cutoff).astype(s.dtype)

        # Apply mask to basis
        basis_filtered = basis * mask

        # Compute SAR
        # Note: We use the filtered basis for projection
        proj_norm = jnp.linalg.norm(get_proj(tensor, basis_filtered))
        tensor_norm = jnp.linalg.norm(tensor) + 1e-6
        return proj_norm / tensor_norm

    u_tgt, s_tgt, _ = jnp.linalg.svd(tau_tgt_2d, full_matrices=False)
    u_src, _, _ = jnp.linalg.svd(tau_src_2d, full_matrices=False)

    sar = get_sar(tau_src_2d, u_tgt, s_tgt, s_tgt.shape[0])

    interaction = u_tgt.T @ u_src

    # shape: [min_dim_c]
    projection_scores = jnp.linalg.norm(interaction, axis=0)

    # 3. Determine Cutoff based on Energy
    # We square the scores to treat them like singular values (energy)
    energies = projection_scores**2
    total_energy = jnp.sum(energies)
    threshold_energy = (sar) * total_energy

    # We must sort energies to find the greedy cutoff,
    # but we need to apply the mask to the ORIGINAL unsorted u_src columns.
    sorted_energies = jnp.sort(energies)[::-1]  # Descending
    cumulative_energy = jnp.cumsum(sorted_energies)

    # Find how many top columns we need to explain 'rank_ratio' of the shared space
    k_cutoff_idx = jnp.searchsorted(cumulative_energy, threshold_energy)

    # Get the score value at that rank index to use as a threshold
    # Handle edge case where k_cutoff_idx is out of bounds
    safe_idx = jnp.clip(k_cutoff_idx, 0, len(sorted_energies) - 1)
    score_threshold_sq = sorted_energies[safe_idx]

    # 4. Create Mask (Keep columns with High Overlap)
    # We keep columns where energy >= threshold (High similarity to B)
    # These are the "Common" parts we want to identify and subtract.
    mask_bool = energies >= score_threshold_sq
    mask = mask_bool.astype(u_tgt.dtype)

    # Apply mask to U_src (zero out unique columns, keep aligned columns)
    u_src_common = u_src * mask[None, :]

    # 5. Project and Subtract
    term = tau_tgt_2d - get_proj(tau_src_2d, u_src_common)

    delta = scaling_coef * sar * term

    return delta.reshape(shape_out).astype(dtype_out)


@dataclasses.dataclass
class DARTConfig(BaseMergeConfig):
    # Arithmetic Settings
    scaling_coef: float = 0.8


class DART(BaseMerging):
    """
    Implementation of Task Arithmetic with Subspace Alignment (JAX Backend).
    Inherits IO and JIT overhead from BaseMerging.
    """

    def __init__(self, cfg: DARTConfig):
        super().__init__(cfg)

    @staticmethod
    def core_arithmetic_fn(
        w_base,
        w_src,
        w_tgt,
        rng_key,
        *,
        scaling_coef,
    ):
        # 1. update-vectors
        tau_src = w_src - w_base
        tau_tgt = w_tgt - w_base

        delta = _greedy_subspace_align_merge(
            tau_src,
            tau_tgt,
            rng_key,
            scaling_coef=scaling_coef,
        )
        return w_base + delta


def main(cfg: DARTConfig) -> None:
    merger = DART(cfg)
    merger.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    jax.config.update("jax_platform_name", "cpu")
    tyro.cli(main)
