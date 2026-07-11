"""Sampling helpers for Tucker pivot search."""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp

from .domains import stage_bounds
from .options import TuckerOptions
from .types import StageContext


def _boundary_samples(
    key: jax.Array, lo: jnp.ndarray, hi: jnp.ndarray, count: int
) -> jnp.ndarray:
    """Sample points biased toward interval endpoints."""
    rem_dim = lo.shape[0]
    if count <= 0:
        return jnp.zeros((0, rem_dim), dtype=jnp.float64)

    key_u, key_side = jax.random.split(key)
    u = jax.random.uniform(key_u, (count, rem_dim), dtype=jnp.float64)
    side = jax.random.bernoulli(key_side, 0.5, shape=(count, rem_dim))

    frac = u**2
    lower = lo + (hi - lo) * frac
    upper = hi - (hi - lo) * frac
    return jnp.where(side, upper, lower)


def _explicit_boundary_points(lo: jnp.ndarray, hi: jnp.ndarray) -> jnp.ndarray:
    """Small deterministic set of corners / face centres for low dimensions."""
    rem_dim = lo.shape[0]
    midpoint = 0.5 * (lo + hi)

    if rem_dim == 1:
        return jnp.stack([lo, hi, midpoint], axis=0)
    if rem_dim == 2:
        corners = jnp.array(list(itertools.product([0, 1], repeat=2)), dtype=jnp.int32)
        corner_pts = jnp.where(corners == 0, lo[None, :], hi[None, :])
        face_pts = jnp.array(
            [
                [lo[0], midpoint[1]],
                [hi[0], midpoint[1]],
                [midpoint[0], lo[1]],
                [midpoint[0], hi[1]],
                midpoint,
            ],
            dtype=jnp.float64,
        )
        return jnp.concatenate([corner_pts, face_pts], axis=0)
    if rem_dim == 3:
        corners = jnp.array(list(itertools.product([0, 1], repeat=3)), dtype=jnp.int32)
        corner_pts = jnp.where(corners == 0, lo[None, :], hi[None, :])
        face_pts = []
        for axis in range(3):
            low_face = midpoint.at[axis].set(lo[axis])
            high_face = midpoint.at[axis].set(hi[axis])
            face_pts.extend([low_face, high_face])
        face_pts.append(midpoint)
        return jnp.concatenate(
            [corner_pts, jnp.stack(face_pts, axis=0)],
            axis=0,
        )
    return jnp.zeros((0, rem_dim), dtype=jnp.float64)


def sample_stage_points(
    key: jax.Array, stage_ctx: StageContext, opts: TuckerOptions
) -> jnp.ndarray:
    """Sample remaining coordinates ``(x_k, ..., x_d)`` for one Tucker stage."""
    lo, hi = stage_bounds(stage_ctx.domain, stage_ctx.stage)
    rem_dim = stage_ctx.rem_dim

    n_boundary = int(round(opts.boundary_bias_fraction * opts.n_starts))
    n_uniform = opts.n_starts - n_boundary

    key_uniform, key_boundary = jax.random.split(key)
    uniform = jax.random.uniform(
        key_uniform,
        (n_uniform, rem_dim),
        minval=lo,
        maxval=hi,
        dtype=jnp.float64,
    )
    boundary = _boundary_samples(key_boundary, lo, hi, n_boundary)
    explicit = _explicit_boundary_points(lo, hi)

    pts = jnp.concatenate([uniform, boundary, explicit], axis=0)
    return jnp.clip(pts, lo, hi)


def expand_samples_over_prefix_indices(
    stage_ctx: StageContext, z_points: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Repeat sample points for every discrete left index.

    At Tucker stage ``k`` there are ``r^{k-1}`` discrete left indices (all
    Cartesian combinations of previous pivot x-values).  This function tiles
    ``z_points`` so that each left index is paired with every sample point.

    Returns
    -------
    alpha_ids : (r^{k-1} * n_samples,) int array
    tiled_z   : (r^{k-1} * n_samples, rem_dim) float array
    """
    z_points = jnp.asarray(z_points, dtype=jnp.float64)
    if z_points.ndim != 2 or z_points.shape[1] != stage_ctx.rem_dim:
        raise ValueError(
            f"z_points must have shape (n, {stage_ctx.rem_dim}), got {z_points.shape}"
        )

    r_prev = int(stage_ctx.prefixes.shape[0])
    alpha_ids = jnp.repeat(jnp.arange(r_prev, dtype=jnp.int32), z_points.shape[0])
    tiled = jnp.tile(z_points, (r_prev, 1))
    return alpha_ids, tiled
