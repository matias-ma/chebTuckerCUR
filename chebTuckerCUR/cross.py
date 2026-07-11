"""Cross-matrix assembly and stable inversion helpers.

The Tucker cross matrix is

    S[j, k] = F(i_j; x_k; y_j),

and we need a stable approximation to ``S^{-1}``.  This module provides
``stable_cross_inverse`` which selects an appropriate solver based on the
condition number and configured backend.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import linalg as jax_linalg

from .options import TuckerOptions
from .types import CrossSolveInfo


def _tikhonov_inverse_numpy(M: np.ndarray, lam: float) -> np.ndarray:
    U, s, Vh = np.linalg.svd(M, full_matrices=False)
    filt = s / (s**2 + lam)
    return (Vh.T * filt) @ U.T


def _tikhonov_inverse_jax(M: jnp.ndarray, lam: float) -> jnp.ndarray:
    U, s, Vh = jnp.linalg.svd(M, full_matrices=False)
    filt = s / (s**2 + lam)
    return (Vh.T * filt) @ U.T


def _qr_inverse_numpy(M: np.ndarray, opts: TuckerOptions) -> tuple[np.ndarray, bool]:
    rank = M.shape[0]
    Q, R = np.linalg.qr(M)
    diag = np.abs(np.diag(R))
    tol = opts.svd_rcond * np.max(diag)
    if rank == 0 or np.any(diag <= tol):
        return np.full_like(M, np.nan), False
    inv_R = np.linalg.solve(R, np.eye(rank, dtype=np.float64))
    W = inv_R @ Q.T
    return W, True


def _qr_inverse_jax(M: jnp.ndarray, opts: TuckerOptions) -> tuple[jnp.ndarray, bool]:
    rank = int(M.shape[0])
    if rank == 0:
        return M, True
    Q, R = jnp.linalg.qr(M)
    diag = jnp.abs(jnp.diag(R))
    tol = float(opts.svd_rcond * jnp.max(diag))
    ok = bool(jnp.all(diag > tol))
    if not ok:
        return jnp.full((rank, rank), jnp.nan, dtype=jnp.float64), False
    inv_R = jax_linalg.solve_triangular(R, jnp.eye(rank, dtype=jnp.float64), lower=False)
    return inv_R @ Q.T, True


def _array_platform(M: jnp.ndarray) -> str:
    try:
        devices = M.devices()
    except AttributeError:
        device = getattr(M, "device", None)
        return getattr(device, "platform", "cpu")
    if not devices:
        return "cpu"
    return getattr(next(iter(devices)), "platform", "cpu")


def _use_jax_cross_solve(M: jnp.ndarray, opts: TuckerOptions) -> bool:
    if opts.cross_solve_backend == "host":
        return False
    if opts.cross_solve_backend == "jax":
        return True
    rank = int(M.shape[0])
    return rank >= opts.cross_solve_jax_min_rank and _array_platform(M) in {"gpu", "tpu"}


def _stable_cross_inverse_host(
    M: jnp.ndarray, opts: TuckerOptions
) -> tuple[jnp.ndarray, CrossSolveInfo]:
    """Host NumPy implementation used for small ranks and CPU runs."""
    rank = int(M.shape[0])
    M_np = np.asarray(M, dtype=np.float64)
    svals = np.linalg.svd(M_np, compute_uv=False)
    sigma_max = float(svals[0])
    sigma_min = float(svals[-1])
    cond = float(jnp.inf if sigma_min == 0.0 else sigma_max / sigma_min)
    effective_rank = int(np.sum(svals > opts.svd_rcond * sigma_max))
    truncated_rank = int(rank - effective_rank)

    eye = np.eye(rank, dtype=np.float64)
    if cond < 1e8:
        W_np = np.linalg.solve(M_np, eye)
        method = "solve"
        accepted = True
        effective_rank = rank
        truncated_rank = 0
    elif opts.cross_solve == "qr" and cond < opts.max_cross_cond:
        W_np, accepted = _qr_inverse_numpy(M_np, opts)
        method = "qr" if accepted else "qr_rejected"
        if not accepted:
            truncated_rank = rank
            effective_rank = 0
    elif cond < opts.max_cross_cond:
        W_np = np.linalg.pinv(M_np, rcond=opts.svd_rcond)
        method = "pinv"
        accepted = True
    elif opts.cross_solve == "tikhonov":
        W_np = _tikhonov_inverse_numpy(M_np, opts.tikhonov_lambda)
        method = "tikhonov"
        accepted = True
    else:
        W_np = np.full_like(M_np, np.nan)
        method = "rejected"
        accepted = False

    info = CrossSolveInfo(
        accepted=accepted,
        method=method,
        cond=cond,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        effective_rank=effective_rank,
        truncated_rank=truncated_rank,
    )
    return jnp.asarray(W_np, dtype=jnp.float64), info


def _stable_cross_inverse_jax(
    M: jnp.ndarray, opts: TuckerOptions
) -> tuple[jnp.ndarray, CrossSolveInfo]:
    """Device JAX implementation for large GPU cross matrices."""
    rank = int(M.shape[0])
    svals = np.asarray(jnp.linalg.svd(M, compute_uv=False), dtype=np.float64)
    sigma_max = float(svals[0])
    sigma_min = float(svals[-1])
    cond = float(np.inf if sigma_min == 0.0 else sigma_max / sigma_min)
    effective_rank = int(np.sum(svals > opts.svd_rcond * sigma_max))
    truncated_rank = int(rank - effective_rank)

    eye = jnp.eye(rank, dtype=jnp.float64)
    if cond < 1e8:
        W = jnp.linalg.solve(M, eye)
        method = "jax_solve"
        accepted = True
        effective_rank = rank
        truncated_rank = 0
    elif opts.cross_solve == "qr" and cond < opts.max_cross_cond:
        W, qr_ok = _qr_inverse_jax(M, opts)
        method = "jax_qr" if qr_ok else "jax_qr_rejected"
        accepted = qr_ok
        if not qr_ok:
            effective_rank = 0
            truncated_rank = rank
    elif cond < opts.max_cross_cond:
        W = jnp.linalg.pinv(M, rtol=opts.svd_rcond)
        method = "jax_pinv"
        accepted = True
    elif opts.cross_solve == "tikhonov":
        W = _tikhonov_inverse_jax(M, opts.tikhonov_lambda)
        method = "jax_tikhonov"
        accepted = True
    else:
        W = jnp.full_like(M, jnp.nan)
        method = "jax_rejected"
        accepted = False

    info = CrossSolveInfo(
        accepted=accepted,
        method=method,
        cond=cond,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        effective_rank=effective_rank,
        truncated_rank=truncated_rank,
    )
    return W, info


def stable_cross_inverse(
    M: jnp.ndarray, opts: TuckerOptions
) -> tuple[jnp.ndarray, CrossSolveInfo]:
    """Compute a stable approximation to ``S^{-1}`` for the Tucker cross matrix.

    Policy:
    - Well-conditioned (cond < 1e8): dense solve against the identity.
    - Moderately conditioned: truncated pseudoinverse or QR inverse.
    - Very ill-conditioned: reject or use Tikhonov, depending on ``opts``.

    Small ranks and CPU arrays use NumPy on the host by default to avoid JAX
    compilation overhead.  Large GPU arrays switch to the JAX path when
    ``opts.cross_solve_backend == "auto"`` and rank exceeds
    ``opts.cross_solve_jax_min_rank``.
    """
    M = jnp.asarray(M, dtype=jnp.float64)
    rank = M.shape[0]
    if M.shape != (rank, rank):
        raise ValueError(f"M must be square, got {M.shape}")
    if rank == 0:
        info = CrossSolveInfo(
            accepted=True, method="empty", cond=1.0, sigma_min=0.0, sigma_max=0.0,
        )
        return M, info

    if _use_jax_cross_solve(M, opts):
        return _stable_cross_inverse_jax(M, opts)
    return _stable_cross_inverse_host(M, opts)
