"""1D Chebyshev core representations and fitting utilities.

Each Tucker factor is a vector-valued one-dimensional function

    G_k : x_k -> R^r,

stored as a ``ChebyshevCoeffCore`` with ``coeffs`` of shape ``(1, r, n)``.
The leading dimension of 1 reflects the scalar input; the trailing dimension
``n`` is the number of Chebyshev coefficients selected to reach ``cheb_tol``.

We also support the more general matrix-valued case

    G_k : x_k -> R^{r_left x r_right}

via the same ``ChebyshevCoeffCore(coeffs[r_left, r_right, n], domain)`` type.
This is used internally by the TT builder and kept here for completeness.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Quadrature / interpolation helpers
# ---------------------------------------------------------------------------

def _chebyshev_lobatto_nodes_and_weights(
    a: float, b: float, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return Chebyshev-Lobatto nodes and Clenshaw-Curtis weights on ``[a, b]``.

    The weights are normalised so they sum to one (i.e. they integrate with
    respect to the uniform measure ``dx / (b - a)``).

    Parameters
    ----------
    a, b : float
        Interval endpoints with ``a < b``.
    n : int
        Number of nodes (must be >= 2).

    Returns
    -------
    x : np.ndarray, shape (n,)
        Lobatto nodes in descending order (from ``b`` to ``a``).
    w : np.ndarray, shape (n,)
        Normalised Clenshaw-Curtis weights.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")

    N = n - 1
    theta = np.pi * np.arange(n) / N
    t = np.cos(theta)
    x = 0.5 * ((b - a) * t + (a + b))
    weights = np.zeros(n, dtype=np.float64)

    if N == 1:
        weights[:] = 1.0
    else:
        interior = np.arange(1, N)
        v = np.ones(N - 1, dtype=np.float64)
        if N % 2 == 0:
            weights[0] = 1.0 / (N * N - 1.0)
            weights[-1] = weights[0]
            for k in range(1, N // 2):
                v -= 2.0 * np.cos(2.0 * k * theta[interior]) / (4.0 * k * k - 1.0)
            v -= np.cos(N * theta[interior]) / (N * N - 1.0)
        else:
            weights[0] = 1.0 / (N * N)
            weights[-1] = weights[0]
            for k in range(1, (N + 1) // 2):
                v -= 2.0 * np.cos(2.0 * k * theta[interior]) / (4.0 * k * k - 1.0)
        weights[interior] = 2.0 * v / N

    # The raw formulas integrate over [-1, 1].  Dividing by 2 converts to
    # the normalised measure dx / (b - a).
    return x, 0.5 * weights


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

def _normalize_entry_layout(entries):
    rows = tuple(tuple(row) for row in entries)
    if len(rows) == 0 or len(rows[0]) == 0:
        raise ValueError("entries must be a non-empty 2D tuple")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("all core rows must have the same width")
    return rows, (len(rows), width)


class ChebfunMatrixCore(eqx.Module):
    """Matrix-valued 1D function represented entry-wise.

    Entry ``entries[a][b]`` approximates the scalar function

        x_k -> G_k[a, b](x_k).
    """

    entries: tuple[tuple[Any, ...], ...]
    domain: tuple[float, float]
    shape: tuple[int, int]

    def __init__(self, entries, domain, shape: tuple[int, int] | None = None):
        rows, inferred_shape = _normalize_entry_layout(entries)
        if shape is not None and shape != inferred_shape:
            raise ValueError(f"shape {shape} does not match entry layout {inferred_shape}")
        self.entries = rows
        self.domain = (float(domain[0]), float(domain[1]))
        self.shape = inferred_shape

    def __call__(self, x):
        x_arr = jnp.asarray(x, dtype=jnp.float64)
        scalar_input = x_arr.ndim == 0

        rows = []
        for entry_row in self.entries:
            vals = [jnp.asarray(entry(x_arr), dtype=jnp.float64) for entry in entry_row]
            rows.append(jnp.stack(vals, axis=-1))

        out = jnp.stack(rows, axis=-2)
        if scalar_input and out.ndim == 3 and out.shape[0] == 1:
            out = out[0]
        return out


class ChebyshevCoeffCore(eqx.Module):
    """Matrix-valued 1D function stored as a Chebyshev coefficient tensor.

    ``coeffs[a, b, n]`` is the coefficient of ``T_n(t(x))`` for the entry
    ``G_k[a, b](x)``, after mapping the physical domain ``[a_k, b_k]`` to
    the reference interval ``[-1, 1]``.

    For Tucker factors ``r_left = 1``, so ``coeffs`` has shape ``(1, r, n)``.
    Calling with a scalar ``x`` returns shape ``(1, r)``; calling with a
    batch of shape ``(B,)`` returns shape ``(B, 1, r)``.
    """

    coeffs: jnp.ndarray
    domain: tuple[float, float]
    shape: tuple[int, int]

    def __init__(self, coeffs, domain):
        coeffs_arr = jnp.asarray(coeffs, dtype=jnp.float64)
        if coeffs_arr.ndim != 3 or coeffs_arr.shape[0] == 0 or coeffs_arr.shape[1] == 0:
            raise ValueError(
                f"coeffs must have shape (r_left, r_right, n_coeffs), got {coeffs_arr.shape}"
            )
        self.coeffs = coeffs_arr
        self.domain = (float(domain[0]), float(domain[1]))
        self.shape = (coeffs_arr.shape[0], coeffs_arr.shape[1])

    def __call__(self, x):
        x_arr = jnp.asarray(x, dtype=jnp.float64)
        scalar_input = x_arr.ndim == 0
        xs = x_arr[None] if scalar_input else x_arr

        a, b = self.domain
        t = (2.0 * xs - (a + b)) / (b - a)

        n_coeffs = self.coeffs.shape[-1]
        result = jnp.broadcast_to(self.coeffs[..., 0], (xs.shape[0],) + self.shape)
        if n_coeffs >= 2:
            t_prev = jnp.ones_like(t)
            t_curr = t
            result = result + t_curr[:, None, None] * self.coeffs[..., 1]
            for k in range(2, n_coeffs):
                t_next = 2.0 * t * t_curr - t_prev
                result = result + t_next[:, None, None] * self.coeffs[..., k]
                t_prev, t_curr = t_curr, t_next

        if scalar_input:
            return result[0]
        return result


# ---------------------------------------------------------------------------
# Coefficient transform
# ---------------------------------------------------------------------------

def _chebyshev_coefficients_from_lobatto_values(values: jnp.ndarray) -> jnp.ndarray:
    """Compute Chebyshev coefficients from values at Lobatto nodes.

    Input is sampled at first-kind Chebyshev-Lobatto nodes

        t_j = cos(pi j / (n - 1)),   j = 0, ..., n - 1.

    The coefficients are obtained via a DCT-I transform implemented with
    ``jnp.fft`` to keep the computation on-device.
    """
    values = jnp.asarray(values, dtype=jnp.float64)
    n = values.shape[0]
    if n < 2:
        raise ValueError(f"need at least two Lobatto values, got {n}")

    reflected = values[-2:0:-1]
    extended = jnp.concatenate([values, reflected], axis=0)
    fft_vals = jnp.fft.rfft(extended, axis=0)
    coeffs = jnp.real(fft_vals[:n]) / (n - 1)
    coeffs = coeffs.at[0].multiply(0.5)
    coeffs = coeffs.at[-1].multiply(0.5)
    return coeffs


# ---------------------------------------------------------------------------
# Core fitting
# ---------------------------------------------------------------------------

def build_coeff_tensor_core(
    entries, domain, tol: float, n_values: tuple[int, ...]
) -> ChebyshevCoeffCore:
    """Build a shared-coefficient matrix core from entry functions."""
    rows, shape = _normalize_entry_layout(entries)

    def matrix_fun(x):
        x_arr = jnp.asarray(x, dtype=jnp.float64)
        rows_out = []
        for row in rows:
            vals = [jnp.asarray(entry(x_arr), dtype=jnp.float64) for entry in row]
            rows_out.append(jnp.stack(vals, axis=-1))
        return jnp.stack(rows_out, axis=-2)

    return build_coeff_tensor_core_from_matrix_function(
        matrix_fun,
        shape=shape,
        domain=domain,
        tol=tol,
        n_values=n_values,
    )


def build_coeff_tensor_core_from_matrix_function(
    matrix_fun,
    *,
    shape: tuple[int, int],
    domain,
    tol: float,
    n_values: tuple[int, ...],
    return_stats: bool = False,
) -> ChebyshevCoeffCore | tuple[ChebyshevCoeffCore, dict[str, int]]:
    """Build a shared-coefficient core from a batched matrix-valued function.

    ``matrix_fun(x_nodes)`` returns the full matrix ``G_k(x_nodes)`` on a
    batch of 1D nodes with shape ``(n,) + shape``.  We fit all entries at
    once by applying the Chebyshev-Lobatto coefficient transform and stop
    when the spectral tail drops below ``tol``.

    Parameters
    ----------
    matrix_fun : callable  (n,) -> (n, r_left, r_right)
        Batched evaluator for the matrix-valued factor.
    shape : (r_left, r_right)
        Output shape of the factor at a single point.
    domain : (a, b)
        Physical interval for this dimension.
    tol : float
        Chebyshev tail stopping threshold.
    n_values : tuple of int
        Increasing node counts to try.
    return_stats : bool
        If True, also return a diagnostics dict.

    Returns
    -------
    ChebyshevCoeffCore (and optionally a stats dict).
    """
    a, b = float(domain[0]), float(domain[1])

    coeff_tensor = None
    sample_point_count = 0
    attempted_n_values = 0
    for n in n_values:
        attempted_n_values += 1
        sample_point_count += int(n)
        t_nodes = np.cos(np.pi * np.arange(n) / (n - 1))
        x_nodes = 0.5 * ((b - a) * t_nodes + (a + b))
        vals = jnp.asarray(
            matrix_fun(jnp.asarray(x_nodes, dtype=jnp.float64)), dtype=jnp.float64
        )
        if vals.shape != (n,) + shape:
            raise ValueError(
                f"matrix_fun must return shape {(n,) + shape} for n={n}, got {vals.shape}"
            )

        coeff_flat = _chebyshev_coefficients_from_lobatto_values(vals.reshape(n, -1))
        coeff_tensor = jnp.transpose(coeff_flat, (1, 0)).reshape(shape + (n,))
        tail_width = min(8, n)
        max_tail = float(jnp.max(jnp.abs(coeff_flat[-tail_width:, :])))
        if max_tail < tol:
            break

    core = ChebyshevCoeffCore(coeff_tensor, domain=(a, b))
    if return_stats:
        return core, {
            "sample_point_count": sample_point_count,
            "attempted_n_values": attempted_n_values,
            "selected_n": int(core.coeffs.shape[-1]),
        }
    return core
