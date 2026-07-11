"""Residual evaluation and cross-matrix construction for Tucker-qTensCUR.

The Tucker cross decomposition at one stage is

    F(i, x, y)  ≈  Y(x) @ U @ C(i, y)

using the qTensCUR naming convention (``C`` carries the discrete/remaining
dependence, ``Y`` carries the stage-variable dependence):

    Y(x)[k]    = F(i_k;  x;    y_k)   -- depends on x only (not i)
    C(i, y)[j] = F(i;    x_j;  y)     -- depends on (i, y) only (not x_j)
    U          = S^{-1},  S[j,k] = F(i_j; x_k; y_j)

The residual is therefore

    e(i, x, y) = F(i, x, y) - Y(x) @ U @ C(i, y).

This differs from the TT residual in that the ``Y`` factor does NOT depend
on the query discrete index ``i``; only the ``C`` factor does.

Note: the code below keeps the local variable/function names ``cvals`` /
``yvals`` (and ``C(x)`` / ``Y(i,y)`` in some in-line comments) from an
earlier version where the roles were named the other way around; the
*mathematics* is unchanged, only the documentation above and in ``build.py``
now uses the corrected ``C`` / ``Y`` naming from the qTensCUR write-up.
"""

from __future__ import annotations

import jax.numpy as jnp


def join_x_y(x, y) -> jnp.ndarray:
    """Concatenate stage variable ``x`` with remaining coordinates ``y``."""
    x_arr = jnp.asarray(x, dtype=jnp.float64)
    y_arr = jnp.asarray(y, dtype=jnp.float64)

    if x_arr.ndim == y_arr.ndim - 1:
        x_arr = x_arr[..., None]
    elif x_arr.ndim == y_arr.ndim:
        if x_arr.shape[-1] != 1:
            x_arr = x_arr[..., None]
    elif x_arr.ndim == 0 and y_arr.ndim == 0:
        x_arr = x_arr[None]
        y_arr = y_arr[None]
    else:
        raise ValueError(f"incompatible x and y shapes: {x_arr.shape}, {y_arr.shape}")

    return jnp.concatenate([x_arr, y_arr], axis=-1)


def build_tucker_cross_matrix(
    F,
    pivot_left_indices: jnp.ndarray,
    pivot_xs: jnp.ndarray,
    pivot_ys: jnp.ndarray,
) -> jnp.ndarray:
    """Build the Tucker cross matrix  ``S[j, k] = F(i_j; x_k; y_j)``.

    Row ``j`` uses the row-pivot pair ``(i_j, y_j)``.
    Column ``k`` uses only the column-pivot x-value ``x_k``.

    This differs from the TT cross matrix where each row also carries an
    x-value.

    Parameters
    ----------
    F : callable  (alpha_ids, z_batch) -> values
        Stage evaluator.
    pivot_left_indices : (r,) int array
        Discrete indices of the row pivots: ``i_0, ..., i_{r-1}``.
    pivot_xs : (r,) float array
        x-values of the column pivots: ``x_0, ..., x_{r-1}``.
    pivot_ys : (r, y_dim) float array
        Trailing coordinates of the row pivots: ``y_0, ..., y_{r-1}``.

    Returns
    -------
    jnp.ndarray, shape (r, r)
    """
    pivot_left_indices = jnp.asarray(pivot_left_indices, dtype=jnp.int32)
    pivot_xs = jnp.asarray(pivot_xs, dtype=jnp.float64)
    pivot_ys = jnp.asarray(pivot_ys, dtype=jnp.float64)

    rank = int(pivot_left_indices.shape[0])
    y_dim = int(pivot_ys.shape[1])

    if rank == 0:
        return jnp.zeros((0, 0), dtype=jnp.float64)

    # j indexes rows (uses i_j, y_j), k indexes columns (uses x_k only).
    j_idx = jnp.repeat(jnp.arange(rank, dtype=jnp.int32), rank)   # (rank^2,)
    k_idx = jnp.tile(jnp.arange(rank, dtype=jnp.int32), rank)     # (rank^2,)

    alpha_batch = pivot_left_indices[j_idx]                        # (rank^2,)
    x_batch = pivot_xs[k_idx]                                      # (rank^2,)
    y_batch = pivot_ys[j_idx]                                      # (rank^2, y_dim)

    z_batch = join_x_y(x_batch, y_batch)                           # (rank^2, 1+y_dim)
    vals = F(alpha_batch, z_batch)                                  # (rank^2,)
    return vals.reshape(rank, rank)


# def tucker_batch_residual_eval(
#     F,
#     pivot_left_indices: jnp.ndarray,
#     pivot_xs: jnp.ndarray,
#     pivot_ys: jnp.ndarray,
#     U: jnp.ndarray,
#     alphas: jnp.ndarray,
#     xs: jnp.ndarray,
#     ys: jnp.ndarray,
# ) -> jnp.ndarray:
#     """Vectorised Tucker residual  ``e(i, x, y) = F(i,x,y) - C(x) @ U @ Y(i,y)``.

#     Tucker cross decomposition:

#         C(x)[k]    = F(i_k, x, y_k)     -- does NOT depend on query ``i``
#         Y(i, y)[j] = F(i, x_j, y)       -- depends on query ``i``, not on ``x_j``

#     Parameters
#     ----------
#     F : callable  (alpha_ids, z_batch) -> values
#         Stage-k evaluator.
#     pivot_left_indices : (r,) int array
#         Discrete indices of the pivot rows: ``i_0, ..., i_{r-1}``.
#     pivot_xs : (r,) float array
#         x-values of the column pivots: ``x_0, ..., x_{r-1}``.
#     pivot_ys : (r, y_dim) float array
#         Trailing coordinates of the pivot rows: ``y_0, ..., y_{r-1}``.
#     U : (r, r) float array
#         Inverse-like factor ``S^{-1}`` of the Tucker cross matrix.
#     alphas : int array
#         Query discrete left indices.
#     xs : float array
#         Query x-values (broadcast-compatible with ``alphas``).
#     ys : float array
#         Query trailing coordinates, shape ``(..., y_dim)``.

#     Returns
#     -------
#     jnp.ndarray, same leading shape as ``alphas``.
#     """
#     pivot_left_indices = jnp.asarray(pivot_left_indices, dtype=jnp.int32)
#     pivot_xs = jnp.asarray(pivot_xs, dtype=jnp.float64)
#     pivot_ys = jnp.asarray(pivot_ys, dtype=jnp.float64)
#     U = jnp.asarray(U, dtype=jnp.float64)
#     alphas_arr = jnp.asarray(alphas, dtype=jnp.int32)
#     xs_arr = jnp.asarray(xs, dtype=jnp.float64)
#     ys_arr = jnp.asarray(ys, dtype=jnp.float64)

#     batch_shape = alphas_arr.shape
#     flat_alpha = alphas_arr.reshape(-1)
#     flat_x = xs_arr.reshape(-1)
#     n = int(flat_alpha.shape[0])
#     rank = int(pivot_left_indices.shape[0])
#     y_dim = int(pivot_ys.shape[1])
#     # y_dim=0 at the last Tucker stage: ys_arr has shape (..., 0).
#     # reshape(-1, 0) raises ZeroDivisionError in JAX, so we special-case it.
#     flat_y = (ys_arr.reshape(n, y_dim) if y_dim > 0
#               else jnp.zeros((n, 0), dtype=jnp.float64))

#     # Base value: F(i, x, y) for each query point.
#     flat_z = join_x_y(flat_x, flat_y)    # (n, 1+y_dim)
#     base = F(flat_alpha, flat_z)          # (n,)

#     if rank == 0:
#         return base.reshape(batch_shape)

#     # C(x)[k] = F(i_k, x, y_k)  — independent of query alpha.
#     # Shape: (n, rank)
#     x_block_c = jnp.broadcast_to(flat_x[:, None, None], (n, rank, 1))
#     y_block_c = jnp.broadcast_to(pivot_ys[None, :, :], (n, rank, y_dim))
#     z_c = jnp.concatenate([x_block_c, y_block_c], axis=-1).reshape(n * rank, 1 + y_dim)
#     alpha_c = jnp.broadcast_to(pivot_left_indices[None, :], (n, rank)).reshape(-1)
#     cvals = F(alpha_c, z_c).reshape(n, rank)    # C(x)[:, k]

#     # Y(i,y)[j] = F(i, x_j, y)  — uses query alpha, fixed pivot x_j.
#     # Shape: (n, rank)
#     x_block_y = jnp.broadcast_to(pivot_xs[None, :, None], (n, rank, 1))
#     y_block_y = jnp.broadcast_to(flat_y[:, None, :], (n, rank, y_dim))
#     z_y = jnp.concatenate([x_block_y, y_block_y], axis=-1).reshape(n * rank, 1 + y_dim)
#     alpha_y = jnp.broadcast_to(flat_alpha[:, None], (n, rank)).reshape(-1)
#     yvals = F(alpha_y, z_y).reshape(n, rank)    # Y(i,y)[:, j]

#     # Approximation.  ``cvals`` (local name "C(x)") is row-type, indexed the
#     # same way as ``Y(x)`` in the qTensCUR write-up; ``yvals`` (local name
#     # "Y(i,y)") is col-type, indexed the same way as ``C(i,y)``.  ``U`` as
#     # returned by stable_cross_inverse has its first index in col-type
#     # space and its second in row-type space (see the note in
#     # ``build._build_tucker_factor``), so the correct contraction pairs
#     # ``yvals``'s (col-type) index with U's first axis and ``cvals``'s
#     # (row-type) index with U's second axis: sum_j sum_k yvals[j] U[j,k] cvals[k].
#     approx = jnp.einsum("nk,jk,nj->n", cvals, U, yvals)
#     return (base - approx).reshape(batch_shape)



def tucker_batch_residual_eval(
    F,
    pivot_left_indices: jnp.ndarray,
    pivot_xs: jnp.ndarray,
    pivot_ys: jnp.ndarray,
    U: jnp.ndarray,
    alphas: jnp.ndarray,
    xs: jnp.ndarray,
    ys: jnp.ndarray,
    chunk_size: int,
) -> jnp.ndarray:
    """Same contract as tucker_batch_residual_eval, but bounds peak memory.
 
    Splits the N candidates into chunks of at most `chunk_size` rows and
    calls the original per-candidate math on each chunk in turn, so no
    intermediate array larger than (chunk_size * rank, ...) is ever
    materialized, regardless of how large N is.
    """
    pivot_left_indices = jnp.asarray(pivot_left_indices, dtype=jnp.int32)
    pivot_xs = jnp.asarray(pivot_xs, dtype=jnp.float64)
    pivot_ys = jnp.asarray(pivot_ys, dtype=jnp.float64)
    U = jnp.asarray(U, dtype=jnp.float64)
    alphas_arr = jnp.asarray(alphas, dtype=jnp.int32)
    xs_arr = jnp.asarray(xs, dtype=jnp.float64)
    ys_arr = jnp.asarray(ys, dtype=jnp.float64)
 
    batch_shape = alphas_arr.shape
    flat_alpha = alphas_arr.reshape(-1)
    flat_x = xs_arr.reshape(-1)
    n_total = int(flat_alpha.shape[0])
    rank = int(pivot_left_indices.shape[0])
    y_dim = int(pivot_ys.shape[1])
    flat_y = (ys_arr.reshape(n_total, y_dim) if y_dim > 0
              else jnp.zeros((n_total, 0), dtype=jnp.float64))
 
    if rank == 0 or n_total == 0:
        flat_z = join_x_y(flat_x, flat_y)
        base = F(flat_alpha, flat_z)
        return base.reshape(batch_shape)
 
    outputs = []
    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        n = end - start
        a_chunk = flat_alpha[start:end]
        x_chunk = flat_x[start:end]
        y_chunk = flat_y[start:end]
 
        flat_z = join_x_y(x_chunk, y_chunk)
        base = F(a_chunk, flat_z)
 
        x_block_c = jnp.broadcast_to(x_chunk[:, None, None], (n, rank, 1))
        y_block_c = jnp.broadcast_to(pivot_ys[None, :, :], (n, rank, y_dim))
        z_c = jnp.concatenate([x_block_c, y_block_c], axis=-1).reshape(n * rank, 1 + y_dim)
        alpha_c = jnp.broadcast_to(pivot_left_indices[None, :], (n, rank)).reshape(-1)
        cvals = F(alpha_c, z_c).reshape(n, rank)
 
        x_block_y = jnp.broadcast_to(pivot_xs[None, :, None], (n, rank, 1))
        y_block_y = jnp.broadcast_to(y_chunk[:, None, :], (n, rank, y_dim))
        z_y = jnp.concatenate([x_block_y, y_block_y], axis=-1).reshape(n * rank, 1 + y_dim)
        alpha_y = jnp.broadcast_to(a_chunk[:, None], (n, rank)).reshape(-1)
        yvals = F(alpha_y, z_y).reshape(n, rank)
 
        approx = jnp.einsum("nk,jk,nj->n", cvals, U, yvals)
        outputs.append(base - approx)
 
    return jnp.concatenate(outputs, axis=0).reshape(batch_shape)