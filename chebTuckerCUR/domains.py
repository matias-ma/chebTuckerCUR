"""Domain helpers for Tucker-qTensCUR construction."""

from __future__ import annotations

import jax.numpy as jnp


def default_domain(d: int) -> jnp.ndarray:
    """Return the default box [-1, 1]^d."""
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    base = jnp.array([-1.0, 1.0], dtype=jnp.float64)
    return jnp.tile(base[None, :], (d, 1))


def normalize_domain(domain, d: int | None = None) -> jnp.ndarray:
    """Normalize domain input to shape (d, 2)."""
    if domain is None:
        if d is None:
            raise ValueError("d must be provided when domain is None")
        return default_domain(d)

    arr = jnp.asarray(domain, dtype=jnp.float64)
    if arr.ndim == 1:
        if arr.shape != (2,):
            raise ValueError(f"1D domain must have shape (2,), got {arr.shape}")
        if d is None:
            raise ValueError("d must be provided when broadcasting a 1D domain")
        arr = jnp.tile(arr[None, :], (d, 1))

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"domain must have shape (d, 2), got {arr.shape}")
    if d is not None and arr.shape[0] != d:
        raise ValueError(f"domain first dimension must be {d}, got {arr.shape[0]}")
    if bool(jnp.any(arr[:, 0] >= arr[:, 1])):
        raise ValueError("each domain interval must satisfy lower < upper")
    return arr


def stage_bounds(domain, stage: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return lower and upper bounds for stage coordinates x_k, ..., x_d."""
    arr = normalize_domain(domain)
    if stage < 1 or stage > arr.shape[0]:
        raise ValueError(f"stage must be in [1, {arr.shape[0]}], got {stage}")
    rem = arr[stage - 1 :]
    return rem[:, 0], rem[:, 1]
