"""Tucker-qTensCUR continuous approximation builder.

This module implements ``build_chebtucker``, which fits a Tucker decomposition

    f̂(x_1,...,x_d) = G ×_1 G_1(x_1) ×_2 ... ×_d G_d(x_d)

via a stage-by-stage qTensCUR algorithm.

Algorithm overview
------------------
At stage k = 1..d:

1. Pivot search.  Run the Tucker cross inner loop to find r pivots
   (alpha_j, x_j, y_j) where:
   - alpha_j  ∈ {0 .. r^{k-1}-1}  — flattened Cartesian multi-index for
                                     the (k-1) already-fixed x-values.
   - x_j      ∈ [a_k, b_k]        — stage-k continuous coordinate.
   - y_j      ∈ R^{d-k}           — remaining continuous coordinates.

2. Cross matrix and inverse.  Build the Tucker cross matrix
   S[j,k] = F_k(alpha_j; x_k; y_j) and compute U = S^{-1}.

3. Factor G_k.  Build a ChebyshevCoeffCore of shape (1, r, n) according to
   ``opts.tucker_update``.  Using the qTensCUR naming
   ``C_k(y)[i,j] = F_k(i; x_j; y)`` and ``Y_k(x)[j] = F_k(i_j; x; y_j)``:

   - "C"  : G_k(x) = U_k Y_k(x)   (U absorbed into the factor; the next
            stage F_{k+1}(x_{>k}) = C_k(x_{>k}) is realised exactly via
            prefix encoding — no further correction needed)
   - "CU" : G_k(x) = Y_k(x)       (no U absorbed; the next stage is
            F_{k+1}(x_{>k}) = C_k(x_{>k}) x_k U_k, a mode-k product with
            U_k, so the Tucker core must be corrected at the end of the
            build by contracting each mode against the corresponding U_k)

4. Prefix update.  Extend the prefix table by the Cartesian product with the
   new pivot x values.  The prefix count grows from r^{k-1} to r^k.

After stage d, evaluate f at all r^d combinations of pivot x values to
obtain the Tucker core tensor G of shape (r,)*d (applying the per-mode U_k
correction afterwards when ``opts.tucker_update == "CU"``).

Tucker vs TT key differences
-----------------------------
Cross matrix: Tucker S[j,k] = F(i_j; x_k; y_j) uses the x from the *column*
pivot, not the row pivot.  This means the Y factor Y(x)[k] = F(i_k, x, y_k)
is independent of the query discrete index i — only the C factor
C(i,y)[j] = F(i, x_j, y) carries i-dependence.

Prefix update: TT selects r accepted row alphas (count stays at r).  Tucker
takes the Cartesian product (count grows as r^k).

Note: an earlier version of this algorithm also L²-orthogonalised the
factors ("qr" / "qr_dense" modes).  This orthogonalisation was unnecessary
for correctness and has been removed; only "C" and "CU" remain.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import chebfunjax as cj
import jax
import jax.numpy as jnp
import numpy as np

from .cores import (
    ChebfunMatrixCore,
    ChebyshevCoeffCore,
    _chebyshev_coefficients_from_lobatto_values,
    _chebyshev_lobatto_nodes_and_weights,
    build_coeff_tensor_core_from_matrix_function,
)
from .cross import stable_cross_inverse
from .domains import normalize_domain, stage_bounds
from .options import TuckerOptions
from .residual import build_tucker_cross_matrix, tucker_batch_residual_eval
from .sampling import expand_samples_over_prefix_indices, sample_stage_points
from .stage_eval import (
    _TuckerStageEval,
    _tucker_update_prefixes,
)
from .tucker import ChebTucker
from .types import StageContext


# ---------------------------------------------------------------------------
# Factor construction — "C" mode
# ---------------------------------------------------------------------------

def _tucker_stage_raw_matrix_fun(
    F,
    pivot_left_indices: jnp.ndarray,
    pivot_ys: jnp.ndarray,
):
    """Return a batched function  x -> Y_k(x) ∈ R^{1 × r}  (no U applied).

        Y_k(x)[k'] = F(i_{k'}, x, y_{k'}).

    ``Y_k`` is bounded by the range of the original function (unlike
    ``U @ Y_k``, whose magnitude can be arbitrarily larger when the cross
    matrix ``S`` is ill-conditioned), so fitting it directly with Chebyshev
    interpolation is numerically well-behaved regardless of how U_k scales.
    Shared by both ``"C"`` and ``"CU"`` modes.

    The returned callable takes a batch of x values of shape ``(n,)`` and
    returns shape ``(n, 1, r)``, matching the convention expected by
    ``build_coeff_tensor_core_from_matrix_function(shape=(1, r), ...)``.
    """
    rank = int(pivot_left_indices.shape[0])
    y_dim = int(pivot_ys.shape[1])
    pivot_left_indices = jnp.asarray(pivot_left_indices, dtype=jnp.int32)
    pivot_ys = jnp.asarray(pivot_ys, dtype=jnp.float64)

    def matrix_fun(x):
        x_arr = jnp.asarray(x, dtype=jnp.float64)
        scalar_input = x_arr.ndim == 0
        xs = x_arr[None] if scalar_input else x_arr
        n = int(xs.shape[0])

        if rank == 0:
            empty = jnp.zeros((n, 1, 0), dtype=jnp.float64)
            return empty[0] if scalar_input else empty

        # Y_k(xs)[n, k'] = F(i_{k'}, xs[n], y_{k'})
        x_block = jnp.broadcast_to(xs[:, None, None], (n, rank, 1))
        y_block = jnp.broadcast_to(pivot_ys[None, :, :], (n, rank, y_dim))
        z_c = jnp.concatenate([x_block, y_block], axis=-1).reshape(n * rank, 1 + y_dim)
        alpha_c = jnp.broadcast_to(pivot_left_indices[None, :], (n, rank)).reshape(-1)
        yvals = F(alpha_c, z_c).reshape(n, rank)   # (n, r)

        out = yvals[:, None, :]    # (n, 1, r)
        return out[0] if scalar_input else out

    return matrix_fun


def _build_tucker_factor(
    F,
    domain_k: tuple[float, float],
    pivot_left_indices: jnp.ndarray,
    pivot_ys: jnp.ndarray,
    U: jnp.ndarray,
    opts: TuckerOptions,
) -> tuple[ChebyshevCoeffCore, dict[str, Any]]:
    """Fit the Tucker factor G_k as a ChebyshevCoeffCore of shape (1, r, n).

    Used for ``"C"`` mode: G_k(x) = U Y_k(x).

    Implementation note: this fits the well-scaled ``Y_k(x)`` with Chebyshev
    interpolation *first*, then applies ``U`` to the resulting coefficient
    tensor (the Chebyshev transform is linear, so this is mathematically
    identical to fitting ``U @ Y_k(x)`` directly).  This was originally
    tried as a fix for the numerical blow-up documented in the README's
    "Known issues" section, on the theory that fitting the *amplified*
    function directly might degrade the interpolation itself.  That theory
    was empirically wrong — the interpolation of ``Y_k`` already converges
    to near machine precision either way, so this reordering changes
    nothing about the blow-up.  It is kept anyway because it lets this
    function share ``_tucker_stage_raw_matrix_fun`` with the "CU" branch in
    ``build.py``, and because it is (trivially) not wrong. The actual
    blow-up mechanism is the cancellation inherent to the final multilinear
    core contraction when factor magnitudes are inflated by an
    ill-conditioned ``U`` — see the README for the full diagnosis and
    ``diagnostics["stage_cross_cond"]`` / ``diagnostics["stage_max_abs_U"]``
    for a way to detect it at build time.

    Returns the fitted core and a stats dict.
    """
    r = int(pivot_left_indices.shape[0])
    raw_matrix_fun = _tucker_stage_raw_matrix_fun(F, pivot_left_indices, pivot_ys)
    raw_core, fit_stats = build_coeff_tensor_core_from_matrix_function(
        raw_matrix_fun,
        shape=(1, r),
        domain=domain_k,
        tol=opts.cheb_tol,
        n_values=opts.coeff_core_n_values,
        return_stats=True,
    )
    function_points = int(fit_stats["sample_point_count"]) * r

    if r == 0:
        core = raw_core
    else:
        # raw_core.coeffs has shape (1, r, n); apply U to the r-axis.
        # G_k(x) = U @ Y_k(x)  (plain matrix-vector product, per the
        # algorithm).  U = S^{-1} as returned by stable_cross_inverse has
        # its first index in the same space as S's column index (the
        # x-pivot type, matching the Tucker core's mode) and its second
        # index in the same space as S's row index (the (i,y)-pivot type,
        # matching Y_k(x)'s index) — so no transpose is needed here:
        # new_coeffs[0, a, :] = sum_b U[a, b] * coeffs[0, b, :].
        U_arr = jnp.asarray(U, dtype=jnp.float64)
        raw_coeffs = raw_core.coeffs[0]                    # (r, n)
        new_coeffs = (U_arr @ raw_coeffs)[None, :, :]      # (1, r, n)
        core = ChebyshevCoeffCore(new_coeffs, domain=raw_core.domain)

    return core, {**fit_stats, "function_points": function_points}


def _build_tucker_factor_chebfun(
    F,
    domain_k: tuple[float, float],
    pivot_left_indices: jnp.ndarray,
    pivot_ys: jnp.ndarray,
    U: jnp.ndarray | None,
    opts: TuckerOptions,
) -> tuple[ChebfunMatrixCore, dict[str, Any]]:
    """Fit the Tucker factor as a row of adaptively-constructed 1D chebfuns.

    This is the ``chebfunjax``-backed analogue of ``_build_tucker_factor`` /
    the "CU" raw-``build_coeff_tensor_core_from_matrix_function`` branch in
    ``build_chebtucker``, mirroring how ``chebttcur``'s
    ``build_tt._build_stage_core`` (the ``"scalar_chebfuns"`` /
    non-``coeff_tensor`` branch) constructs each scalar entry of a stage core
    as its own ``cj.chebfun(...)`` rather than sampling a fixed
    ``coeff_core_n_values`` ladder of Chebyshev-Lobatto grids.

    Because the Tucker factor ``Y_k(x)`` (and its ``U``-transformed version
    in ``"C"`` mode) does not depend on the discrete prefix index ``alpha``
    (see the README's "Tucker vs TT cross matrix" note), the resulting core
    has shape ``(1, r)`` — a single row of ``r`` independent scalar chebfuns,
    one per pivot column. If ``U`` is given, each output entry ``a`` is the
    already-CUR-transformed scalar function
    ``x -> sum_b U[a, b] * Y_k(x)[b]`` (matching ``"C"`` mode, and mirroring
    ``build_tt._build_core_entry_function``'s convention of applying the
    cross-inverse to *raw sampled values* before fitting, not to already-fit
    coefficients). If ``U`` is ``None``, each output entry ``b`` is the raw
    ``Y_k(x)[b] = F(i_b, x, y_b)`` (matching ``"CU"`` mode).

    Returns the fitted ``ChebfunMatrixCore`` and a stats dict with the same
    ``"function_points"`` / ``"representation"`` keys used elsewhere.
    """
    r = int(pivot_left_indices.shape[0])
    a_k, b_k = domain_k
    raw_matrix_fun = _tucker_stage_raw_matrix_fun(F, pivot_left_indices, pivot_ys)
    eval_counter = {"function_points": 0}

    def counted_raw_matrix_fun(x):
        x_arr = jnp.asarray(x, dtype=jnp.float64)
        n = 1 if x_arr.ndim == 0 else int(x_arr.shape[0])
        eval_counter["function_points"] += n * r
        return raw_matrix_fun(x)

    if U is not None:
        U_arr = jnp.asarray(U, dtype=jnp.float64)

        def make_entry(a: int):
            def entry(x):
                y = counted_raw_matrix_fun(x)          # (..., 1, r)
                y = jnp.squeeze(y, axis=-2)             # (..., r)
                return y @ U_arr[a, :]

            return entry

        col_funs = [make_entry(a) for a in range(r)]
    else:

        def make_raw_entry(b: int):
            def entry(x):
                y = counted_raw_matrix_fun(x)           # (..., 1, r)
                y = jnp.squeeze(y, axis=-2)              # (..., r)
                return y[..., b]

            return entry

        col_funs = [make_raw_entry(b) for b in range(r)]

    cheb_row = tuple(cj.chebfun(entry, domain=[a_k, b_k], eps=opts.cheb_tol, splitting=True) for entry in col_funs)
    core = ChebfunMatrixCore(entries=(cheb_row,), domain=(float(a_k), float(b_k)))
    return core, {
        "representation": "scalar_chebfuns",
        "function_points": int(eval_counter["function_points"]),
    }


# ---------------------------------------------------------------------------
# Pivot search
# ---------------------------------------------------------------------------

def _tucker_select_pivots(
    key: jax.Array,
    F,
    stage_ctx: StageContext,
    target_rank: int,
    opts: TuckerOptions,
) -> list[tuple[int, float, jnp.ndarray, float]]:
    """Search for up to ``target_rank`` Tucker pivots at one stage.

    Each pivot is a tuple ``(alpha, x, y, residual_value)`` where:
    - ``alpha`` ∈ {0 .. r^{k-1}-1}  — discrete left index
    - ``x``     ∈ [a_k, b_k]        — stage-k continuous coordinate (scalar)
    - ``y``     ∈ R^{d-k}           — trailing continuous coordinates
    - ``residual_value``            — signed residual at this point

    Strategy
    --------
    Start with rank 0 (U empty).  Loop up to ``target_rank`` times:
      1. Sample ``(alpha, z=(x,y))`` candidates via the sampler.
      2. Evaluate the Tucker residual for every candidate.
      3. Accept the candidate with the largest |residual|, provided it
         passes the absolute tolerance and conditioning guards.
      4. Rebuild S and U = S^{-1} with the new pivot included.

    Returns
    -------
    list of (alpha, x, y_arr, residual_value) tuples, length ≤ target_rank.
    """
    pivots: list[tuple[int, float, jnp.ndarray, float]] = []
    pivot_left_indices = jnp.array([], dtype=jnp.int32)
    pivot_xs = jnp.array([], dtype=jnp.float64)
    pivot_ys = jnp.zeros((0, max(stage_ctx.rem_dim - 1, 0)), dtype=jnp.float64)
    U = jnp.zeros((0, 0), dtype=jnp.float64)

    for _ in range(target_rank):
        key, subkey = jax.random.split(key)
        z0 = sample_stage_points(subkey, stage_ctx, opts)
        alpha_ids, z_candidates = expand_samples_over_prefix_indices(stage_ctx, z0)

        z_flat = jnp.asarray(z_candidates, dtype=jnp.float64)
        alpha_flat = jnp.asarray(alpha_ids, dtype=jnp.int32)

        if len(pivots) == 0:
            # Rank 0: residual equals F itself.
            resid_vals = F(alpha_flat, z_flat)
        else:
            resid_vals = tucker_batch_residual_eval(
                F,
                pivot_left_indices,
                pivot_xs,
                pivot_ys,
                U,
                alpha_flat,
                z_flat[:, 0],
                z_flat[:, 1:],
                opts.residual_chunk_size,
            )
            # resid_vals = tucker_batch_residual_eval(
            #     F,
            #     pivot_left_indices,
            #     pivot_xs,
            #     pivot_ys,
            #     U,
            #     alpha_flat,
            #     z_flat[:, 0],
            #     z_flat[:, 1:],
            # )

        abs_resid = jnp.abs(resid_vals)
        best_idx = int(jnp.argmax(abs_resid))
        best_abs = float(abs_resid[best_idx])

        if best_abs < opts.pivot_abs_tol:
            break  # Residual below threshold; stop.

        best_alpha = int(alpha_flat[best_idx])
        best_z = z_flat[best_idx]
        best_x = float(best_z[0])
        best_y = best_z[1:]

        # Propose extending S and U.
        new_left = jnp.concatenate([pivot_left_indices, jnp.array([best_alpha], dtype=jnp.int32)])
        new_xs = jnp.concatenate([pivot_xs, jnp.array([best_x], dtype=jnp.float64)])
        new_ys = jnp.concatenate([pivot_ys, best_y[None, :]], axis=0)

        S_new = build_tucker_cross_matrix(F, new_left, new_xs, new_ys)
        W_new, info = stable_cross_inverse(S_new, opts)
        if not info.accepted or info.cond > opts.max_cross_cond:
            continue  # Cross matrix too ill-conditioned; skip.

        pivot_left_indices = new_left
        pivot_xs = new_xs
        pivot_ys = new_ys
        U = W_new

        resid_val = float(best_abs) * float(jnp.sign(resid_vals[best_idx]))
        pivots.append((best_alpha, best_x, best_y, resid_val))

    return pivots


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_chebtucker(
    f,
    opts: TuckerOptions,
    d: int | None = None,
) -> ChebTucker:
    """Fit a Tucker decomposition via Tucker-qTensCUR.

    Parameters
    ----------
    f : callable  (batch, d) -> (batch,)
        Target function.  Must accept batched float64 input arrays of shape
        ``(n, d)`` and return shape ``(n,)``.
    opts : TuckerOptions
        All build options.  Key field: ``opts.tucker_update`` selects the
        stage update mode (``"C"`` | ``"CU"``; see ``TuckerOptions``).
    d : int or None
        Dimension.  Inferred from ``opts.domain`` when not supplied.

    Returns
    -------
    ChebTucker
        Fitted Tucker approximation ready for evaluation.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from chebtucker import build_chebtucker, TuckerOptions
    >>>
    >>> def f(x):   # x: (n, 3)
    ...     return jnp.sin(x[:, 0]) * jnp.cos(x[:, 1]) * jnp.exp(-x[:, 2])
    >>>
    >>> opts = TuckerOptions(max_rank=4, tucker_update="C")
    >>> approx = build_chebtucker(f, opts, d=3)
    >>> approx(jnp.array([0.1, 0.2, 0.3]))
    """
    if d is None:
        if opts.domain is None:
            raise ValueError("d must be provided when opts.domain is None")
        d = int(normalize_domain(opts.domain).shape[0])

    domain = opts.resolved_domain(d)
    # ``target_rank`` is the pivot-search budget offered to *every* stage.
    # It must stay fixed across stages: a stage converging early (because
    # its own factor happens to be exactly low-rank) says nothing about the
    # intrinsic rank needed by a *later* stage/mode, so a smaller
    # ``r_actual`` discovered at one stage must not shrink the search
    # budget handed to subsequent stages (see the "Rank update" comment
    # below for the bug this used to cause).
    target_rank = int(opts.ranks[0]) if opts.ranks is not None else int(opts.max_rank)

    key = jax.random.PRNGKey(opts.random_seed)

    # Stage 1 starts with one empty prefix (left rank = 1).
    prefixes = jnp.zeros((1, 0), dtype=jnp.float64)
    factors: list[ChebyshevCoeffCore] = []
    all_pivot_xs: list[jnp.ndarray] = []
    stage_ranks: list[int] = []   # actual rank accepted at each stage
    diagnostics: dict[str, Any] = {
        "tucker_update": opts.tucker_update,
        "tucker_rank": target_rank,
        "factor_representation": opts.factor_representation,
        "stage_times_sec": [],
        "stage_pivot_counts": [],
        "stage_cross_cond": [],
        "stage_max_abs_U": [],
        "total_function_value_points": 0,
    }

    # For "CU" mode: F_{k+1}(x_{>k}) = C_k(x_{>k}) x_k U_k is realised on the
    # prefix grid *without* applying U_k (the prefix trick only realises
    # C_k(x_{>k}) exactly), so each stage's U_k must be applied to the final
    # Tucker core after the fact, one mode at a time.  Not used in "C" mode,
    # where U_k is absorbed directly into the factor G_k instead.
    transfer_matrices: list[jnp.ndarray] = []

    for stage in range(1, d + 1):
        stage_start = perf_counter()
        stage_ctx = StageContext(stage=stage, d=d, prefixes=prefixes, domain=domain)

        # Build the stage function F_k.  In both "C" and "CU" modes,
        # F_{k+1}(x_{>k}) = C_k(x_{>k}) is realised exactly via the
        # Cartesian-product prefix table, so every stage simply re-evaluates
        # the original function f at the current prefixes — no chaining or
        # coefficient absorption is needed here.
        F = _TuckerStageEval(f, stage_ctx)

        # ------------------------------------------------------------------
        # Pivot search.
        # ------------------------------------------------------------------
        key, pivot_key = jax.random.split(key)
        pivots = _tucker_select_pivots(pivot_key, F, stage_ctx, target_rank, opts)

        if len(pivots) == 0:
            raise RuntimeError(
                f"Tucker pivot search found no acceptable pivot at stage {stage}. "
                "Consider increasing opts.n_starts or decreasing opts.pivot_abs_tol."
            )

        pivot_left_indices = jnp.array([p[0] for p in pivots], dtype=jnp.int32)
        pivot_xs_raw = jnp.array([p[1] for p in pivots], dtype=jnp.float64)
        pivot_ys = jnp.stack([p[2] for p in pivots], axis=0)   # (r_actual, y_dim)
        r_actual = len(pivots)

        # Rebuild S and compute U = S^{-1} from the accepted pivots.
        S = build_tucker_cross_matrix(F, pivot_left_indices, pivot_xs_raw, pivot_ys)
        U, solve_info = stable_cross_inverse(S, opts)

        # ------------------------------------------------------------------
        # Build the Tucker factor G_k.
        # ------------------------------------------------------------------
        a_k, b_k = float(domain[stage - 1, 0]), float(domain[stage - 1, 1])
        domain_k = (a_k, b_k)
        fp = 0

        use_chebfun = opts.factor_representation == "scalar_chebfuns"

        if opts.tucker_update == "C":
            # G_k(x) = U Y_k(x);  F_{k+1}(x_{>k}) = C_k(x_{>k}) is realised
            # exactly via prefix encoding — no core correction is needed.
            if use_chebfun:
                factor_core, factor_stats = _build_tucker_factor_chebfun(
                    F, domain_k, pivot_left_indices, pivot_ys, U, opts
                )
            else:
                factor_core, factor_stats = _build_tucker_factor(
                    F, domain_k, pivot_left_indices, pivot_ys, U, opts
                )
            fp = int(factor_stats.get("function_points", 0))

        elif opts.tucker_update == "CU":
            # G_k(x) = Y_k(x)  (no U absorbed);
            # F_{k+1}(x_{>k}) = C_k(x_{>k}) x_k U_k, a mode-k product with
            # U_k that is NOT realised by the prefix trick (which only ever
            # realises the untransformed C_k(x_{>k})).  We therefore stash
            # U_k here and apply it as a per-mode correction to the Tucker
            # core once the raw core has been evaluated on the prefix grid,
            # after all d stages (see the "CU core correction" block below).
            if use_chebfun:
                factor_core, raw_stats = _build_tucker_factor_chebfun(
                    F, domain_k, pivot_left_indices, pivot_ys, None, opts
                )
                fp = int(raw_stats.get("function_points", 0))
            else:
                raw_matrix_fun = _tucker_stage_raw_matrix_fun(F, pivot_left_indices, pivot_ys)
                factor_core, raw_stats = build_coeff_tensor_core_from_matrix_function(
                    raw_matrix_fun,
                    shape=(1, r_actual),
                    domain=domain_k,
                    tol=opts.cheb_tol,
                    n_values=opts.coeff_core_n_values,
                    return_stats=True,
                )
                fp = int(raw_stats.get("sample_point_count", 0)) * r_actual
            # Stash U_k^T; applied to core mode k (0-indexed: stage-1) below.
            transfer_matrices.append(U.T)

        else:
            raise ValueError(f"unsupported tucker_update {opts.tucker_update!r}")

        factors.append(factor_core)
        all_pivot_xs.append(pivot_xs_raw)

        # ------------------------------------------------------------------
        # Rank bookkeeping: this stage's own factor/core axis uses
        # ``r_actual`` (whatever the pivot search actually found — Tucker
        # ranks may legitimately differ from mode to mode).  Unlike an
        # earlier version of this code, ``r_actual`` is *not* propagated
        # forward as a shrunk budget for subsequent stages: a stage
        # converging early (e.g. because its factor happens to be exactly
        # low-rank, or because its own pivot search saturated the
        # conditioning guard) says nothing about the rank a *different*
        # stage/mode needs. Cascading the smaller value forward previously
        # starved later stages of pivot-search budget, silently forcing
        # them to stop after only ``r_actual`` rounds regardless of
        # whether their own residual had actually converged — which in
        # turn made the greedy search settle on whichever few candidates
        # happened to look best early on (often a numerically "easy"
        # discrete slice), rather than the pivots the later stage's own
        # richer structure actually required. ``target_rank`` (the search
        # budget) therefore stays fixed at the originally requested rank
        # for every stage; only the per-stage *result* is allowed to vary.
        # ------------------------------------------------------------------
        stage_ranks.append(r_actual)

        # Update prefixes: Cartesian product with the accepted pivot x-values.
        prefixes = _tucker_update_prefixes(prefixes, pivot_xs_raw)

        elapsed = perf_counter() - stage_start
        diagnostics["stage_times_sec"].append(elapsed)
        diagnostics["stage_pivot_counts"].append(r_actual)
        diagnostics["stage_cross_cond"].append(float(solve_info.cond))
        diagnostics["stage_max_abs_U"].append(float(jnp.max(jnp.abs(U))) if U.size else 0.0)
        diagnostics["total_function_value_points"] += fp

    # --------------------------------------------------------------------------
    # Tucker core.
    #
    # After d stages, ``prefixes`` has shape (r^d, d): the full Cartesian
    # product of all pivot x-values.  In both modes the raw core is obtained
    # by evaluating f directly on this prefix (pivot) grid:
    #
    # - "C"  : G_k already absorbed U_k, so the raw core needs no correction.
    # - "CU" : G_k did NOT absorb U_k, so we must correct the raw core by
    #          contracting each mode k against the corresponding U_k
    #          (stashed, transposed, in ``transfer_matrices``), matching
    #          F_{k+1}(x_{>k}) = C_k(x_{>k}) x_k U_k at every stage.
    # --------------------------------------------------------------------------
    core_start = perf_counter()
    n_core = int(prefixes.shape[0])   # r^d

    core_vals = f(prefixes)       # (r^d,)
    core_arr = core_vals.reshape(tuple(stage_ranks))

    if opts.tucker_update == "CU" and transfer_matrices:
        # Apply the accumulated U_k corrections along each mode:
        #   G' = G ×_1 U_1 ×_2 U_2 ... ×_d U_d
        # Each T[k] = U_k^T has shape (r_k, r_k) where r_k = stage_ranks[k].
        for k, T in enumerate(transfer_matrices):
            core_arr = jnp.moveaxis(
                jnp.tensordot(T, jnp.moveaxis(core_arr, k, 0), axes=([1], [0])),
                0, k
            )

    diagnostics["core_function_points"] = n_core
    diagnostics["total_function_value_points"] += n_core
    diagnostics["core_build_time_sec"] = perf_counter() - core_start

    return ChebTucker(
        core=core_arr,
        factors=tuple(factors),
        domain=domain,
        ranks=tuple(stage_ranks),
        diagnostics=diagnostics,
        stage_pivot_xs=tuple(all_pivot_xs),
    )
