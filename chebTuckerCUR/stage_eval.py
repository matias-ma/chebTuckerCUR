"""Stage evaluation and Tucker-specific recursion helpers.

The core idea: stage ``k`` works with a function

    F_k(alpha, z_rem) = f(prefixes[alpha], z_rem),

where ``prefixes[alpha]`` stores the Cartesian-product combination of pivot
x-values ``(x_1^{j_1}, ..., x_{k-1}^{j_{k-1}})`` for multi-index ``alpha``,
and ``z_rem = (x_k, ..., x_d)`` are the remaining continuous variables.

This avoids nested Python closures: future stages are evaluated by direct
calls to the original ``f`` with stored prefixes.

Naming convention (matches ``qTensCUR(F_k, r) -> C_k(y), U_k, Y_k(x)``):

    C_k(y)[i, j] = F_k(i; x_j; y)   -- depends on the prefix index i and y
    Y_k(x)[j]    = F_k(i_j; x; y_j) -- depends only on the stage variable x

Tucker-specific helpers
-----------------------
``_TuckerStageEval``
    Simple wrapper: F(alpha, z_rem) = f(prefixes[alpha], z_rem).  Used for
    every stage in both the ``"C"`` and ``"CU"`` update modes: the next
    stage's function ``F_{k+1}(x_{>k}) = C_k(x_{>k})`` is realised exactly
    via the Cartesian-product prefix table, with no coefficient mixing
    required (any ``U_k`` correction needed by the ``"CU"`` mode is applied
    to the Tucker core at the very end of the build instead — see
    ``build.py``).

``_tucker_update_prefixes``
    Cartesian-product prefix extension:
    new_prefixes[I * r + j] = (prefixes[I], pivot_xs[j]).

Note: an earlier version of this module also implemented an L²-orthogonalised
update mode (``"qr"`` / ``"qr_dense"``, via classes ``TuckerQRProjectedFunction``
and ``TuckerQRDenseProjectedFunction``).  That orthogonalisation was not
necessary for correctness and has been removed; only the ``"C"`` and ``"CU"``
modes remain.
"""

from __future__ import annotations

import jax.numpy as jnp

from .types import StageContext


# ---------------------------------------------------------------------------
# Generic prefix-based stage evaluation (used by "cur_y" and "cur_uy" modes)
# ---------------------------------------------------------------------------

def _normalize_z_rem(rem_dim: int, z_rem) -> jnp.ndarray:
    """Normalize the remaining coordinates to have trailing dimension ``rem_dim``."""
    z = jnp.asarray(z_rem, dtype=jnp.float64)
    if rem_dim == 1 and (z.ndim == 0 or (z.ndim >= 1 and z.shape[-1] != 1)):
        z = z[..., None]
    if z.shape[-1] != rem_dim:
        raise ValueError(f"z_rem must end in dimension {rem_dim}, got {z.shape}")
    return z


def assemble_full_inputs(stage_ctx: StageContext, alpha, z_rem) -> jnp.ndarray:
    """Assemble the original ``d``-dimensional input from ``alpha`` and ``z_rem``.

    Returns ``x_full = (prefixes[alpha], z_rem)`` with shape ``(..., d)``.
    """
    z = _normalize_z_rem(stage_ctx.rem_dim, z_rem)
    alpha_arr = jnp.asarray(alpha, dtype=jnp.int32)
    lead_shape = z.shape[:-1]
    alpha_arr = jnp.broadcast_to(alpha_arr, lead_shape)
    prefix = stage_ctx.prefixes[alpha_arr]
    return jnp.concatenate([prefix, z], axis=-1)


def make_stage_eval(f, stage_ctx: StageContext):
    """Return the stage evaluator ``F_k(alpha, z_rem)``.

    The returned callable supports both scalar and batched ``alpha`` and
    ``z_rem`` as long as their leading shapes broadcast.
    """
    def F(alpha, z_rem):
        return f(assemble_full_inputs(stage_ctx, alpha, z_rem))
    return F


# ---------------------------------------------------------------------------
# Tucker stage evaluator
# ---------------------------------------------------------------------------

class _TuckerStageEval:
    """Simple wrapper: ``F(alpha, z_rem) = f(prefixes[alpha], z_rem)``.

    Used for the ``"cur_y"`` and ``"cur_uy"`` Tucker update modes where no
    coefficient mixing is needed, and always for stage 1 in every mode.

    Parameters
    ----------
    f : callable  (batch, d) -> (batch,)
        Original target function.
    stage_ctx : StageContext
        Current stage context carrying ``prefixes`` and dimension info.
    """

    def __init__(self, f, stage_ctx: StageContext) -> None:
        self._f = f
        self._prefixes = stage_ctx.prefixes
        self._rem_dim = stage_ctx.rem_dim
        self._d = stage_ctx.d
        self.left_rank = int(stage_ctx.prefixes.shape[0])
        self.function_point_cost = 1

    def __call__(self, alpha, z_rem):
        alpha_arr = jnp.asarray(alpha, dtype=jnp.int32).reshape(-1)
        z_arr = jnp.asarray(z_rem, dtype=jnp.float64)
        if z_arr.ndim == 1:
            z_arr = z_arr[None, :]
        prefix = self._prefixes[alpha_arr]
        full = jnp.concatenate([prefix, z_arr], axis=-1)
        return self._f(full)


# ---------------------------------------------------------------------------
# Tucker prefix update — Cartesian product
# ---------------------------------------------------------------------------

def _tucker_update_prefixes(
    prefixes: jnp.ndarray,
    pivot_xs: jnp.ndarray,
) -> jnp.ndarray:
    """Extend prefixes by the Cartesian product with the new pivot x-values.

    In TT, the new prefixes are obtained by *selecting* ``r`` accepted row
    alphas:  ``prefixes_next[beta] = (prefixes[alpha_beta], x_beta)``.

    In Tucker, every old prefix is extended with every new pivot x:

        new_prefixes[I * r + j] = (prefixes[I], pivot_xs[j])

    so the prefix count grows from ``r^{k-1}`` to ``r^k``.

    Parameters
    ----------
    prefixes : (r^{k-1}, k-1) float array
        Current stage prefixes.
    pivot_xs : (r,) float array
        The ``r`` pivot x-values selected at this stage.

    Returns
    -------
    jnp.ndarray, shape (r^{k-1} * r, k)
    """
    prefixes = jnp.asarray(prefixes, dtype=jnp.float64)
    pivot_xs = jnp.asarray(pivot_xs, dtype=jnp.float64)
    r_prev = int(prefixes.shape[0])
    r = int(pivot_xs.shape[0])

    prefixes_rep = jnp.repeat(prefixes, r, axis=0)      # (r_prev * r, k-1)
    xs_tile = jnp.tile(pivot_xs, r_prev)[:, None]       # (r_prev * r, 1)
    return jnp.concatenate([prefixes_rep, xs_tile], axis=1)  # (r_prev * r, k)

