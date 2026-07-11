"""
Utilities for exploiting a continuous Tucker decomposition

    f_hat(x_1,...,x_d) = G x_1 G_1(x_1) x_2 ... x_d G_d(x_d)

to compute integrals and 2D marginals in O(d) 1-D operations + small core
contractions, instead of evaluating f_hat on a dense d-dimensional grid.
Works for general d.

This version assumes each Tucker factor G_i is stored the way
``build.py``'s ``_build_tucker_factor_chebfun`` produces it when
``opts.factor_representation == "scalar_chebfuns"``: a ``ChebfunMatrixCore``
of shape (1, r_i) whose single row is a tuple of r_i ``chebfunjax``
``Chebfun`` objects, one per Tucker pivot column. Since each column is
already an adaptively-fit chebfun, it knows how to integrate itself
(``.sum()``) and evaluate itself (vectorized via ``jax.vmap``, per the
chebfunjax README), so this module no longer needs to implement its own
Gauss-Legendre quadrature or per-point evaluation fallback -- it just
orchestrates those per-factor chebfunjax calls across the d modes.

Assumed interface of `approx`:
    approx.core     : array of shape (r_1, ..., r_d)
    approx.factors  : list of length d; approx.factors[i] is a
                      ChebfunMatrixCore with
                        - .entries[0]  : tuple of r_i cj.chebfun objects
                                         (the factor's single row)
                        - .domain      : (a_i, b_i), the domain every
                                         chebfun in that row was built on
"""

import jax
import jax.numpy as jnp


def _factor_chebfuns(Gi):
    """Return the tuple of r_i scalar chebfuns making up factor G_i.

    ``Gi`` is a ``ChebfunMatrixCore`` of shape (1, r_i): a single row of
    chebfuns produced by ``build.py``'s ``_build_tucker_factor_chebfun``.
    Its ``.entries`` is a length-1 tuple of rows; ``entries[0]`` is the
    row itself -- a length-r_i tuple of ``cj.chebfun`` objects.
    """
    if not hasattr(Gi, "entries"):
        raise TypeError(
            "tucker_integral/tucker_marginal_2d now expect chebfunjax-backed "
            "factors (ChebfunMatrixCore, i.e. opts.factor_representation == "
            "'scalar_chebfuns'), but got a factor with no `.entries` "
            f"attribute ({type(Gi)!r}). Rebuild with "
            "factor_representation='scalar_chebfuns', or use the old "
            "quadrature-based tools for coeff_tensor-style factors."
        )
    return Gi.entries[0]


def _factor_integrals(Gi):
    """Vector of integrals of each column chebfun in factor G_i.

    wi[k] = integral of the k-th chebfun in the row, obtained directly via
    chebfunjax's own ``.sum()`` (adaptive Clenshaw-Curtis on the Chebyshev
    coefficients already stored in the chebfun) rather than re-quadraturing
    by hand.
    """
    return jnp.array([c.sum() for c in _factor_chebfuns(Gi)])


def _factor_eval(Gi, x):
    """Evaluate every column chebfun of factor G_i at points x, shape (n,).

    Returns shape (n, r_i). Each chebfun is called directly under
    ``jax.vmap`` (chebfunjax functions are vmap-friendly per the README's
    "Batched evaluation" example), so no manual batching/fallback logic is
    needed here.
    """
    x = jnp.asarray(x)
    cols = [jax.vmap(c)(x) for c in _factor_chebfuns(Gi)]
    return jnp.stack(cols, axis=-1)


def tucker_integral(approx):
    """
    Integral of the Tucker approximant over the product of each factor's
    own chebfun domain (i.e. the domain each G_i was built on).

    Parameters
    ----------
    approx : object with .core (r_1 x ... x r_d) and .factors (list of d
             ChebfunMatrixCore objects, one row of chebfuns each)

    Returns
    -------
    float
    """
    core = jnp.asarray(approx.core)
    d = core.ndim
    assert len(approx.factors) == d, "dimension mismatch"

    result = core
    for Gi in approx.factors:
        wi = _factor_integrals(Gi)                       # (r_i,)
        result = jnp.tensordot(wi, result, axes=(0, 0))   # always contract axis 0
    return float(result)


def tucker_marginal_2d(approx, dims, grid_points=200):
    """
    2D marginal g(x_i, x_j) = int f_hat dx_(rest), for dims=(i, j).

    Parameters
    ----------
    approx      : Tucker approximant (see tucker_integral)
    dims        : (i, j) 0-indexed dimensions to keep
    grid_points : resolution per axis of the returned grid; the plotting
                  range for the two kept dims comes straight from their own
                  factors' ``.domain`` (no separate `domains` argument
                  needed anymore)

    Returns
    -------
    X, Y, Z : 2D arrays (meshgrid, indexing='ij') ready for contourf/pcolormesh
    """
    core = jnp.asarray(approx.core)
    d = core.ndim
    i, j = dims
    other = [m for m in range(d) if m not in dims]

    # Move the two kept modes to the end, integrate out the rest one at a time.
    order = other + [i, j]
    reduced = jnp.transpose(core, order)

    for m in other:
        wi = _factor_integrals(approx.factors[m])
        reduced = jnp.tensordot(wi, reduced, axes=(0, 0))
    # reduced now has shape (r_i, r_j)

    Gi, Gj = approx.factors[i], approx.factors[j]
    a_i, b_i = Gi.domain
    a_j, b_j = Gj.domain
    xi = jnp.linspace(a_i, b_i, grid_points)
    xj = jnp.linspace(a_j, b_j, grid_points)

    Gi_vals = _factor_eval(Gi, xi)   # (grid_points, r_i)
    Gj_vals = _factor_eval(Gj, xj)   # (grid_points, r_j)

    Z = jnp.einsum('kl,pk,ql->pq', reduced, Gi_vals, Gj_vals)
    X, Y = jnp.meshgrid(xi, xj, indexing='ij')
    return X, Y, Z


def plot_tucker_marginal_2d(approx, dims, grid_points=200, ax=None, cmap='viridis',
                             **contourf_kwargs):
    """Convenience wrapper: compute and plot the 2D marginal."""
    import matplotlib.pyplot as plt

    X, Y, Z = tucker_marginal_2d(approx, dims, grid_points)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(X, Y, Z, levels=40, cmap=cmap, **contourf_kwargs)
    plt.colorbar(cs, ax=ax)
    ax.set_xlabel(f"$x_{{{dims[0]+1}}}$")
    ax.set_ylabel(f"$x_{{{dims[1]+1}}}$")
    ax.set_title("2D marginal")
    return ax


# if __name__ == "__main__":
#     # ---- example usage for d = 5, opts.factor_representation="scalar_chebfuns" ----
#     # I = tucker_integral(approx)
#     #
#     # plot_tucker_marginal_2d(approx, dims=(0, 2))
#     # import matplotlib.pyplot as plt
#     # plt.show()
#     pass