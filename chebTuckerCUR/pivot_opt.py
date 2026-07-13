"""Local refinement of sampled Tucker pivot candidates.

At each greedy round, ``build._tucker_select_pivots`` wants

    max_{alpha, z} |e_k(alpha, z)|

over the discrete left index ``alpha`` (a flattened prefix multi-index) and
the remaining continuous box variable ``z = (x_k, ..., x_d)``. Without this
module, that max is approximated purely by ``jnp.argmax`` over one batch of
sampled candidates (see ``sampling.sample_stage_points`` /
``expand_samples_over_prefix_indices``) — the accepted pivot is only ever as
good as that one batch, with no local ascent around the sampled maximizer.

``refine_candidates`` adds exactly that ascent step: a few projected
Adam/GD updates of each candidate's *continuous* ``z``, holding its discrete
``alpha`` fixed (``alpha`` isn't continuous, so it is never touched), moving
each ``z`` toward a nearby local maximizer of ``e_k(alpha, ·)^2`` before the
final arg-max over the (now-refined) candidate pool is taken. This mirrors
the ``pivot_opt.optimize_candidates`` local-search step used by this
codebase's TT sibling, scoped down to Tucker's simpler one-pivot-per-round
loop (Tucker has no block/validation/CPLU machinery to port).

This is a heuristic local search, not an exact pivoting procedure: gradient
ascent on a smooth surrogate of ``|e_k|`` can still land on a poor local
maximum, walk to a domain boundary, or (rarely) walk away from an
already-good sample entirely. The last case is why every candidate keeps
whichever of {initial sample, refined endpoint} has the larger ``|e_k|``,
never the refined point unconditionally.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .domains import stage_bounds
from .options import TuckerOptions
from .residual import tucker_batch_residual_eval
from .types import StageContext


def _residual_batch(
    F,
    pivot_left_indices: jnp.ndarray,
    pivot_xs: jnp.ndarray,
    pivot_ys: jnp.ndarray,
    U: jnp.ndarray,
    alpha_ids: jnp.ndarray,
    z: jnp.ndarray,
    chunk_size: int,
) -> jnp.ndarray:
    """Evaluate e_k(alpha, z) for the current (possibly empty) pivot set.

    Matches the branch already used in ``build._tucker_select_pivots``: with
    no pivots yet, e_1 = F itself; otherwise the closed-form cross residual.
    """
    if int(pivot_left_indices.shape[0]) == 0:
        return F(alpha_ids, z)
    return tucker_batch_residual_eval(
        F,
        pivot_left_indices,
        pivot_xs,
        pivot_ys,
        U,
        alpha_ids,
        z[:, 0],
        z[:, 1:],
        chunk_size,
    )


def _surrogate_loss(
    z: jnp.ndarray,
    F,
    pivot_left_indices: jnp.ndarray,
    pivot_xs: jnp.ndarray,
    pivot_ys: jnp.ndarray,
    U: jnp.ndarray,
    alpha_ids: jnp.ndarray,
    chunk_size: int,
    objective: str,
    eps: float,
) -> jnp.ndarray:
    """Smooth surrogate whose ascent makes |e_k| large; summed over the batch
    since each candidate's z is independent and jax.grad differentiates the
    scalar sum, giving one gradient per-candidate in a single backward pass.
    """
    resid = _residual_batch(F, pivot_left_indices, pivot_xs, pivot_ys, U, alpha_ids, z, chunk_size)
    if objective == "square":
        return jnp.sum(-(resid**2))
    return jnp.sum(-jnp.log(resid**2 + eps**2))


def refine_candidates(
    F,
    stage_ctx: StageContext,
    alpha_ids: jnp.ndarray,
    z0: jnp.ndarray,
    pivot_left_indices: jnp.ndarray,
    pivot_xs: jnp.ndarray,
    pivot_ys: jnp.ndarray,
    U: jnp.ndarray,
    opts: TuckerOptions,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Locally refine sampled candidates and return their residual values.

    Parameters
    ----------
    F : callable  (alpha_ids, z) -> values
        The current stage function (``_TuckerStageEval``).
    stage_ctx : StageContext
        Used only for the stage's continuous box bounds (``stage_bounds``).
    alpha_ids : (N,) int array
        Discrete prefix index for each candidate. Fixed throughout — never
        differentiated or moved.
    z0 : (N, rem_dim) float array
        Sampled starting points, i.e. ``sampling.expand_samples_over_prefix_indices``'s
        tiled continuous coordinates.
    pivot_left_indices, pivot_xs, pivot_ys, U :
        The already-accepted pivots and current cross-inverse for this
        stage (all length-0 arrays before the first pivot of a stage).
    opts : TuckerOptions
        ``opts.gd_steps == 0`` disables refinement and reproduces the
        original sample-and-argmax behaviour exactly (same residual values
        as evaluating ``z0`` directly, at the same cost as before).

    Returns
    -------
    z_final : (N, rem_dim) float array
        For each candidate, whichever of {``z0``, refined endpoint} has the
        larger ``|e_k|``.
    resid_final : (N,) array
        ``e_k`` evaluated at ``z_final`` (signed, matching the
        ``jnp.sign(resid_vals[best_idx])`` convention used downstream).
    """
    chunk_size = opts.residual_chunk_size
    resid0 = _residual_batch(F, pivot_left_indices, pivot_xs, pivot_ys, U, alpha_ids, z0, chunk_size)

    if opts.gd_steps <= 0:
        return z0, resid0

    lo, hi = stage_bounds(stage_ctx.domain, stage_ctx.stage)

    def loss_fn(z):
        return _surrogate_loss(
            z, F, pivot_left_indices, pivot_xs, pivot_ys, U, alpha_ids, chunk_size,
            opts.optimizer_objective, opts.optimizer_eps,
        )

    grad_fn = jax.grad(loss_fn)

    def step_body(step, carry):
        z, m, v = carry
        g = grad_fn(z)
        if opts.gd_method == "gd":
            z_new = jnp.clip(z - opts.gd_lr * g, lo, hi)
            return z_new, m, v
        beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g**2)
        mhat = m / (1.0 - beta1**step)
        vhat = v / (1.0 - beta2**step)
        z_new = jnp.clip(z - opts.gd_lr * mhat / (jnp.sqrt(vhat) + adam_eps), lo, hi)
        return z_new, m, v

    z_final, _, _ = jax.lax.fori_loop(
        1, opts.gd_steps + 1, step_body, (z0, jnp.zeros_like(z0), jnp.zeros_like(z0))
    )
    resid_final = _residual_batch(F, pivot_left_indices, pivot_xs, pivot_ys, U, alpha_ids, z_final, chunk_size)

    # Ascent on the smooth surrogate can occasionally walk away from an
    # already-good sampled start (surrogate optimum != |e_k| optimum, or the
    # step simply overshoots); keep whichever is actually better in |e_k|.
    keep_initial = jnp.abs(resid0) > jnp.abs(resid_final)
    z_out = jnp.where(keep_initial[:, None], z0, z_final)
    resid_out = jnp.where(keep_initial, resid0, resid_final)
    return z_out, resid_out
