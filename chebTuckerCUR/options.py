"""Options for Tucker-qTensCUR construction."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .domains import normalize_domain


@dataclass(frozen=True)
class TuckerOptions:
    """Configuration for Tucker-qTensCUR construction.

    Parameters
    ----------
    domain : array-like of shape (d, 2) or None
        Physical domain box.  ``None`` defaults to ``[-1, 1]^d``.
    max_rank : int
        Tucker rank ``r`` when ``ranks`` is not set.
    ranks : tuple of int or None
        Per-stage ranks.  When provided, ``ranks[0]`` is used as the uniform
        Tucker rank (all stages share the same rank in the current algorithm).
    factor_representation : {"scalar_chebfuns", "coeff_tensor"}
        How each Tucker factor $G_k$ is fit and stored.

        ``"scalar_chebfuns"`` (default)
            Each of the ``r`` columns of ``Y_k`` (or ``U @ Y_k`` in ``"C"``
            mode) is fit independently as its own adaptively-resolved
            ``chebfunjax`` chebfun via ``build._build_tucker_factor_chebfun``,
            producing a ``cores.ChebfunMatrixCore`` of shape ``(1, r)``. Each
            column gets its own degree, resolved to the chebfun's own
            adaptive tolerance, rather than sharing one degree across all
            ``r`` columns. This is required by ``compute_tools.py``
            (``tucker_integral`` / ``tucker_marginal_2d``), which calls the
            underlying chebfun objects' own ``.sum()`` and vectorized
            evaluation directly.
        ``"coeff_tensor"``
            Each factor is fit as a single shared-coefficient
            ``cores.ChebyshevCoeffCore`` of shape ``(1, r, n)`` via
            ``cores.build_coeff_tensor_core_from_matrix_function``: all
            ``r`` columns are sampled together on a Chebyshev-Lobatto grid
            whose size is chosen from ``coeff_core_n_values`` using a
            shared, heuristic tail-decay check against ``cheb_tol`` (not an
            adaptive per-column chebfun). Kept for cross-checking and for
            avoiding the ``chebfunjax`` per-column construction overhead when
            ``compute_tools.py``-style calculus on the factors isn't needed.
    tucker_update : {"C", "CU"}
        Stage update mode, following ``qTensCUR(F_k, r) -> C_k(y), U_k, Y_k(x)``
        where ``C_k(y)[i, j] = F_k(i; x_j; y)`` (depends on the discrete
        prefix index ``i`` and the remaining coordinates ``y``) and
        ``Y_k(x)[j] = F_k(i_j; x; y_j)`` (depends only on the stage
        variable ``x``):

        ``"C"``
            ``G_k(x_k) = U_k Y_k(x_k)``.  ``U_k`` is absorbed directly into
            the factor.  The next-stage function is realised exactly,
            ``F_{k+1}(x_{>k}) = C_k(x_{>k})``, via prefix encoding — no
            further correction is needed.  This is the default and
            recommended mode ("recurse on C").
        ``"CU"``
            ``G_k(x_k) = Y_k(x_k)`` with no inversion absorbed into the
            factor.  The next-stage function is
            ``F_{k+1}(x_{>k}) = C_k(x_{>k}) x_k U_k`` (mode-``k`` product
            with ``U_k``), so the Tucker core must be corrected at the end
            of the build by contracting each mode against the corresponding
            ``U_k``.  Kept mainly for cross-checking against ``"C"``; in
            exact arithmetic both modes give the same approximant.

        Note: an earlier version of this algorithm additionally
        L²-orthogonalised the factors (``"qr"`` / ``"qr_dense"`` modes).
        This was unnecessary for the correctness of the approximation and
        has been removed.
    tol : float
        Target approximation tolerance (used by adaptive-rank extensions).
    cheb_tol : float
        Chebyshev coefficient tail threshold for determining the polynomial
        degree of each Tucker factor.
    coeff_core_n_values : tuple of int
        Increasing sequence of Chebyshev node counts tried when fitting each
        Tucker factor until ``cheb_tol`` is satisfied.
    n_starts : int
        Number of random starting points per pivot search iteration.
    boundary_bias_fraction : float in [0, 1]
        Fraction of ``n_starts`` points biased toward domain boundaries.
    residual_chunk_size : int
        Maximum number of candidate points processed at once inside
        ``tucker_batch_residual_eval`` during pivot search.  The full
        candidate batch has size ``N = r_prev * n_samples`` (which grows as
        ``r^(k-1)`` across stages), and for each candidate the residual
        evaluation internally builds arrays of size ``N * rank``.  Chunking
        bounds the peak size of those arrays to ``residual_chunk_size *
        rank`` regardless of how large ``N`` gets, trading a small amount of
        loop overhead for bounded memory.  Total function evaluations are
        unchanged — this does not reduce the underlying ``r^(k-1)``-scaling
        cost of Tucker's Cartesian prefix growth, only the peak memory
        needed to compute it.  Lower this if you hit out-of-memory errors at
        high ``max_rank``; raise it if memory isn't a constraint, for
        slightly less loop overhead.
    pivot_abs_tol : float
        Absolute residual threshold below which a pivot candidate is rejected.
    max_cross_cond : float
        Maximum allowed condition number for the Tucker cross matrix ``S``.
    cross_solve : {"solve", "qr", "svd", "tikhonov"}
        Strategy used by ``stable_cross_inverse`` when ``S`` is ill-conditioned.
    cross_solve_backend : {"auto", "host", "jax"}
        Whether to compute ``S^{-1}`` on the host (NumPy) or on-device (JAX).
    cross_solve_jax_min_rank : int
        Minimum rank at which the JAX backend is preferred in ``"auto"`` mode.
    svd_rcond : float
        Relative singular-value cutoff for pseudoinverse and QR solves.
    tikhonov_lambda : float
        Regularisation parameter used when ``cross_solve == "tikhonov"``.
    random_seed : int
        Base seed for all JAX PRNG calls during construction.
    gd_steps : int
        Number of local-refinement (gradient ascent) steps applied to each
        round's sampled candidates before taking the arg-max, targeting
        ``max_{alpha, z} |e_k(alpha, z)|`` instead of accepting the sampled
        arg-max as-is (see ``pivot_opt.refine_candidates``).  ``0`` (the
        default) disables refinement and reproduces the original
        sample-and-argmax behaviour exactly.  The discrete index ``alpha``
        is never refined (it isn't continuous); only the continuous
        ``z = (x_k, ..., x_d)`` part of each candidate is moved.
    gd_lr : float
        Step size used by the local refinement optimizer.
    gd_method : {"gd", "adam"}
        ``"gd"`` takes plain gradient-ascent steps (projected back into the
        stage box after each step); ``"adam"`` uses Adam moment estimates
        (also projected).
    optimizer_objective : {"square", "log"}
        Smooth surrogate maximized in place of ``|e_k|`` during refinement.
        ``"square"`` maximizes ``e_k^2``.  ``"log"`` maximizes
        ``log(e_k^2 + optimizer_eps^2)``, which grows more slowly far from a
        root of ``e_k`` and can behave better when residual magnitudes vary
        by orders of magnitude across the sampled candidates.
    optimizer_eps : float
        Smoothing constant used by the ``"log"`` objective.
    """

    domain: jnp.ndarray | None = None
    max_rank: int = 8
    ranks: tuple[int, ...] | None = None
    factor_representation: str = "scalar_chebfuns"
    tucker_update: str = "C"
    tol: float = 1e-8
    cheb_tol: float = 1e-8
    coeff_core_n_values: tuple[int, ...] = (17, 33, 65, 129, 257)
    n_starts: int = 256
    boundary_bias_fraction: float = 0.25
    residual_chunk_size: int = 100_000
    pivot_abs_tol: float = 1e-14
    max_cross_cond: float = 1e10
    cross_solve: str = "svd"
    cross_solve_backend: str = "auto"
    cross_solve_jax_min_rank: int = 128
    svd_rcond: float = 1e-10
    tikhonov_lambda: float = 1e-14
    random_seed: int = 0
    gd_steps: int = 0
    gd_lr: float = 0.05
    gd_method: str = "adam"
    optimizer_objective: str = "square"
    optimizer_eps: float = 1e-12

    def __post_init__(self) -> None:
        if self.domain is not None:
            object.__setattr__(self, "domain", normalize_domain(self.domain))
        if self.max_rank <= 0:
            raise ValueError("max_rank must be positive")
        if self.ranks is not None and len(self.ranks) == 0:
            raise ValueError("ranks must be non-empty when provided")
        if self.factor_representation not in {"scalar_chebfuns", "coeff_tensor"}:
            raise ValueError(
                f"unsupported factor_representation {self.factor_representation!r}"
            )
        if self.tucker_update not in {"C", "CU"}:
            raise ValueError(f"unsupported tucker_update {self.tucker_update!r}")
        if len(self.coeff_core_n_values) == 0 or any(n < 2 for n in self.coeff_core_n_values):
            raise ValueError("coeff_core_n_values must contain integers >= 2")
        if self.n_starts <= 0:
            raise ValueError("n_starts must be positive")
        if not 0.0 <= self.boundary_bias_fraction <= 1.0:
            raise ValueError("boundary_bias_fraction must lie in [0, 1]")
        if self.residual_chunk_size <= 0:
            raise ValueError("residual_chunk_size must be positive")
        if self.max_cross_cond <= 1.0:
            raise ValueError("max_cross_cond must exceed 1")
        if self.cross_solve not in {"solve", "qr", "svd", "tikhonov"}:
            raise ValueError(f"unsupported cross_solve {self.cross_solve!r}")
        if self.cross_solve_backend not in {"auto", "host", "jax"}:
            raise ValueError(f"unsupported cross_solve_backend {self.cross_solve_backend!r}")
        if self.cross_solve_jax_min_rank < 1:
            raise ValueError("cross_solve_jax_min_rank must be positive")
        if self.svd_rcond <= 0.0:
            raise ValueError("svd_rcond must be positive")
        if self.tikhonov_lambda <= 0.0:
            raise ValueError("tikhonov_lambda must be positive")
        if self.gd_steps < 0:
            raise ValueError("gd_steps must be non-negative")
        if self.gd_lr <= 0.0:
            raise ValueError("gd_lr must be positive")
        if self.gd_method not in {"gd", "adam"}:
            raise ValueError(f"unsupported gd_method {self.gd_method!r}")
        if self.optimizer_objective not in {"square", "log"}:
            raise ValueError(f"unsupported optimizer_objective {self.optimizer_objective!r}")
        if self.optimizer_eps <= 0.0:
            raise ValueError("optimizer_eps must be positive")

    def resolved_domain(self, d: int) -> jnp.ndarray:
        """Resolve the working domain for dimension ``d``."""
        return normalize_domain(self.domain, d=d)