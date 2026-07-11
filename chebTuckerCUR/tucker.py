"""Continuous Tucker-format approximation object.

A Tucker decomposition of a d-dimensional function is

    f̂(x_1,...,x_d) = G ×_1 G_1(x_1) ×_2 G_2(x_2) ... ×_d G_d(x_d)
                   = Σ_{j_1,...,j_d} G[j_1,...,j_d] G_1(x_1)[j_1] ... G_d(x_d)[j_d]

where:
- ``G ∈ R^{r × ... × r}``  is the Tucker core tensor (shape ``(r,)*d``)
- each factor ``G_k : x_k → R^r`` is a scalar-to-vector function, stored as a
  ``ChebyshevCoeffCore`` with ``coeffs`` of shape ``(1, r, n_cheb_k)``

Evaluation
----------
The multilinear contraction is computed dimension by dimension, contracting the
Tucker core with each factor vector in turn.  For a single point ``x``:

    v = G                                   # shape (r,)*d
    for k in 0..d-1:
        v = jnp.tensordot(v, G_k(x_k), axes=([0], [0]))   # contracts first axis

After d such contractions ``v`` is a scalar.

For a batch ``x`` of shape ``(B, d)`` the same loop applies dimension by
dimension and the result has shape ``(B,)``.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from .cores import ChebyshevCoeffCore  # noqa: F401 — re-exported for convenience


class ChebTucker(eqx.Module):
    """Continuous Tucker approximation.

    Parameters
    ----------
    core : jnp.ndarray, shape (r,)*d
        Dense Tucker core tensor G.
    factors : tuple of ChebyshevCoeffCore
        One factor per dimension.  Factor k has ``coeffs`` of shape
        ``(1, r, n_cheb_k)``; when called with a scalar ``x`` it returns a
        vector of shape ``(1, r)`` (the leading 1-dimension is squeezed to
        ``(r,)`` during contraction).
    domain : jnp.ndarray, shape (d, 2)
        Physical box ``[a_k, b_k]`` for each dimension k.
    ranks : tuple of int
        Tucker ranks; currently a uniform value ``(r,) * d``.
    diagnostics : dict
        Build-time diagnostics forwarded from ``build_chebtucker``.
    stage_pivot_xs : tuple of jnp.ndarray or None
        Pivot x-values selected at each stage.  ``stage_pivot_xs[k-1]`` is
        an array of shape ``(r_k,)`` holding the column-pivot x-coordinates
        chosen at stage ``k``.  Used to compute the lebesgue-type matrix
        ``Qhat_k[j, ell] = G_k(x_k^ell)[j]`` and its inverse norms.
    """

    core: jnp.ndarray
    factors: tuple[Any, ...]
    domain: jnp.ndarray
    ranks: tuple[int, ...]
    diagnostics: dict[str, Any]
    stage_pivot_xs: tuple[Any, ...]

    def __init__(
        self,
        core,
        factors,
        domain,
        ranks: tuple[int, ...] | None = None,
        diagnostics: dict[str, Any] | None = None,
        stage_pivot_xs: tuple | None = None,
    ) -> None:
        core_arr = jnp.asarray(core, dtype=jnp.float64)
        factors = tuple(factors)
        d = len(factors)
        if d == 0:
            raise ValueError("ChebTucker requires at least one factor")
        if core_arr.ndim != d:
            raise ValueError(
                f"core must be a rank-{d} tensor, got shape {core_arr.shape}"
            )

        domain_arr = jnp.asarray(domain, dtype=jnp.float64)
        if domain_arr.shape != (d, 2):
            raise ValueError(
                f"domain must have shape ({d}, 2), got {domain_arr.shape}"
            )

        # Verify each factor's rank matches the corresponding core axis.
        # Ranks may differ across stages when the pivot search found fewer
        # pivots than requested at some stage (rank-reduction path).
        for k, fac in enumerate(factors):
            fac_r = fac.shape[1]
            core_r_k = core_arr.shape[k]
            if fac_r != core_r_k:
                raise ValueError(
                    f"factor {k} right rank {fac_r} does not match "
                    f"core axis {k} size {core_r_k}"
                )

        if ranks is None:
            ranks = tuple(core_arr.shape)
        if len(ranks) != d:
            raise ValueError(f"ranks must have length {d}, got {len(ranks)}")

        self.core = core_arr
        self.factors = factors
        self.domain = domain_arr
        self.ranks = tuple(int(r_) for r_ in ranks)
        self.diagnostics = {} if diagnostics is None else dict(diagnostics)
        if stage_pivot_xs is None:
            self.stage_pivot_xs = tuple()
        else:
            self.stage_pivot_xs = tuple(
                jnp.asarray(xs, dtype=jnp.float64) for xs in stage_pivot_xs
            )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def __call__(self, x):
        """Evaluate the Tucker approximation at point(s) x.

        Parameters
        ----------
        x : array-like, shape (d,) or (batch, d)

        Returns
        -------
        jnp.ndarray  scalar or shape (batch,)
        """
        x_arr = jnp.asarray(x, dtype=jnp.float64)
        scalar_input = x_arr.ndim == 1
        if scalar_input:
            x_arr = x_arr[None, :]
        B, d = x_arr.shape
        if d != len(self.factors):
            raise ValueError(
                f"x must have {len(self.factors)} coordinates, got {d}"
            )

        # Evaluate all factor vectors at once.
        # factor_vecs[k] has shape (B, r).
        factor_vecs = []
        for k, fac in enumerate(self.factors):
            # ChebyshevCoeffCore(1, r, n): calling with a batch of shape (B,)
            # returns (B, 1, r).  We squeeze the middle axis to get (B, r).
            fv = fac(x_arr[:, k])       # (B, 1, r)
            factor_vecs.append(fv[:, 0, :])  # (B, r)

        # Contract the Tucker core with each factor in turn.
        # After k contractions the working tensor has shape
        # (B, r, r, ..., r) with (d - k) trailing r-axes.
        v = jnp.broadcast_to(self.core[None, ...], (B,) + self.core.shape)
        for k in range(d):
            fv_k = factor_vecs[k]   # (B, r)
            # "bi..." contracts shared index i; remaining "..." axes of v and
            # the batch axis b are preserved.
            v = jnp.einsum("bi...,bi->b...", v, fv_k)
        # v is now shape (B,).

        out = v.reshape(B)
        if scalar_input:
            return out[0]
        return out

    # ------------------------------------------------------------------
    # Lebesgue-type diagnostics
    # ------------------------------------------------------------------

    def qhat_matrix(self, stage: int) -> jnp.ndarray:
        r"""Evaluate factor ``G_k`` at its own column-pivot x-values.

        Defines the square matrix

            Qhat_k[j, ell] = G_k(x_k^{(ell)})[j],
                for j, ell in {0, ..., r-1}.

        where ``x_k^{(ell)}`` are the column-pivot x-values chosen at
        stage ``k`` during Tucker-qTensCUR.

        Parameters
        ----------
        stage : int
            Stage index ``k`` (1-based, in ``[1, d]``).

        Returns
        -------
        jnp.ndarray, shape (r, r)
            The matrix ``Qhat_k``.

        Raises
        ------
        ValueError
            If ``stage`` is out of range or ``stage_pivot_xs`` was not
            stored (e.g. the object was constructed without pivot data).
        """
        d = len(self.factors)
        if not (1 <= stage <= d):
            raise ValueError(f"stage must be in [1, {d}], got {stage}")
        if len(self.stage_pivot_xs) == 0:
            raise ValueError(
                "stage_pivot_xs is empty — rebuild with build_chebtucker "
                "(version that stores pivot x-values)."
            )
        if stage > len(self.stage_pivot_xs):
            raise ValueError(
                f"stage_pivot_xs has only {len(self.stage_pivot_xs)} entries; "
                f"cannot access stage {stage}."
            )

        xs = self.stage_pivot_xs[stage - 1]   # (r,)
        fac = self.factors[stage - 1]          # ChebyshevCoeffCore, shape (1, r, n)

        # Evaluate factor at all r pivot x-values.
        # fac(xs) returns shape (r, 1, r): index [ell, 0, j] = G_k(x^ell)[j].
        fac_vals = fac(xs)                     # (r, 1, r)
        # Qhat[j, ell] = fac_vals[ell, 0, j]  →  transpose last two axes.
        Qhat = fac_vals[:, 0, :].T             # (r, r)
        return Qhat

    def qhat_inv_inf_norms(self) -> list[float]:
        r"""Compute ``||Qhat_k^{-1}||_inf`` for every stage ``k``.

        The infinity norm of a matrix is the maximum absolute row sum:

            ||A||_inf = max_i  sum_j |A[i, j]|.

        A large value indicates that the pivot x-values are poorly
        conditioned for the factor at that stage (analogous to a large
        Lebesgue constant).

        Returns
        -------
        list of float, length d
            ``norms[k-1]`` is ``||Qhat_k^{-1}||_inf``.

        Raises
        ------
        ValueError
            If ``stage_pivot_xs`` was not stored.
        numpy.linalg.LinAlgError
            If any ``Qhat_k`` is singular (use ``qhat_matrix`` to
            inspect the individual matrices in that case).
        """
        import numpy as np

        norms = []
        for stage in range(1, len(self.factors) + 1):
            Qhat = jnp.asarray(self.qhat_matrix(stage), dtype=jnp.float64)
            Qhat_np = np.array(Qhat)
            Qhat_inv = np.linalg.inv(Qhat_np)
            # Infinity norm: max absolute row sum.
            inf_norm = float(np.max(np.sum(np.abs(Qhat_inv), axis=1)))
            norms.append(inf_norm)
        return norms

    def _call_single(self, x):
        """Scalar evaluation without batch overhead (useful for debugging)."""
        x_arr = jnp.asarray(x, dtype=jnp.float64)
        if x_arr.ndim != 1 or x_arr.shape[0] != len(self.factors):
            raise ValueError(
                f"x must be 1-D with {len(self.factors)} coordinates, got {x_arr.shape}"
            )
        d = len(self.factors)
        v = self.core                        # (r,)*d
        for k in range(d):
            fv = self.factors[k](x_arr[k])  # shape (1, r) for scalar input
            g = fv[0] if fv.ndim == 2 else fv  # (r,)
            v = jnp.tensordot(v, g, axes=([0], [0]))
        return v  # scalar