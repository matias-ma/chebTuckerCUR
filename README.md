# chebtucker

Continuous Tucker decomposition of multivariate functions via **Tucker-qTensCUR**.

## What it does

Given $f : [-1,1]^d \to \mathbb{R}$, `chebtucker` produces

$$\hat{f}(x_1,\ldots,x_d) = \mathcal{G} \times_1 G_1(x_1) \times_2 \cdots \times_d G_d(x_d)$$

where $\mathcal{G} \in \mathbb{R}^{r \times \cdots \times r}$ is a dense core tensor and each
factor $G_k : \mathbb{R} \to \mathbb{R}^r$ is a vector of univariate functions, fit stage by
stage via qTensCUR (cross approximation with pivots chosen greedily by sampling).

## Algorithm

### qTensCUR (one stage)

Given $F \in \qTens(n_1,\ldots,n_q,[-1,1]^p)$ and target rank $r$, greedily pick $r$ pivots
$(\boldsymbol{i}_k,x_k,\boldsymbol{y}_k)$ by maximizing $|e_k|$ under the residual update

$$e_{k+1}(\boldsymbol{i},x,\boldsymbol{y}) = e_k(\boldsymbol{i},x,\boldsymbol{y}) - \frac{e_k(\boldsymbol{i},x_k,\boldsymbol{y})}{e_k(\boldsymbol{i}_k,x_k,\boldsymbol{y}_k)}e_k(\boldsymbol{i}_k,x,\boldsymbol{y}_k),$$

then form the cross matrix $[S]_{j,k}=F(\boldsymbol{i}_j;x_k;\boldsymbol{y}_j)$, $U:=S^{-1}$, and

$$[C(\boldsymbol{y})]_{\boldsymbol{i},k}=F(\boldsymbol{i};x_k;\boldsymbol{y}),\qquad [Y(x)]_k=F(\boldsymbol{i}_k;x;\boldsymbol{y}_k),$$

giving $F(x,\boldsymbol{y})\approx C(\boldsymbol{y})\times_{q+1}U\,Y(x)$.

**Naming.** $C$ carries the discrete/remaining dependence, $Y$ the stage-variable
dependence — this matches the qTensCUR write-up but is the *opposite* of an older
version of this codebase, and `residual.py` still has some internal variable names
(`cvals`/`yvals`) left over from that older convention. The math is correct; only
those two local names are swapped relative to what they compute.

**How pivots are actually found (`_tucker_select_pivots` in `build.py`).** The
$\arg\max$ over $(\boldsymbol{i},x,\boldsymbol{y})$ is approximated by sampling, not
continuous optimization, at each of up to $r$ greedy rounds:

1. `sampling.sample_stage_points` draws `opts.n_starts` points $(x,\boldsymbol{y})$ — a mix
   of uniform points, boundary-biased points, and a few deterministic corners/face
   midpoints (only for remaining dimension $\le 3$).
2. `sampling.expand_samples_over_prefix_indices` pairs every sample with every existing
   discrete prefix, giving a candidate pool of size `n_starts * r_prev`.
3. `residual.tucker_batch_residual_eval` evaluates $e_k$ at every candidate (or $F_k$
   itself for the first pivot of a stage) via the closed-form cross residual
   $e = F - Y\,U\,C$, not by literally chaining the rank-1 update above. The candidate
   with the largest $|e_k|$ is proposed.
4. A proposed pivot is accepted only if augmenting $S$ keeps `stable_cross_inverse`
   happy (`info.accepted` and $\mathrm{cond}(S)\le$ `opts.max_cross_cond`); otherwise it's
   dropped and the round retried with a fresh sample.
5. A stage's search stops early — for the rest of that stage — once the best *sampled*
   $|e_k|$ falls below `opts.pivot_abs_tol`. This is a statement about the current
   sample, not the true continuous residual, so `diagnostics["stage_pivot_counts"]` can
   under-shoot `opts.max_rank` even when the function genuinely needs more terms,
   especially at large `rem_dim` or for thin/oscillatory residuals.
6. `types.CandidateBatch` has fields for optimizer-based refinement
   (`optimizer_value_function_points`, etc.) but no such refinement is wired up —
   selection is pure sample-and-argmax, and those fields stay `0`.

### Tucker-qTensCUR (the full sweep)

`build_chebtucker` sweeps $k=1,\ldots,d$: run qTensCUR on $F_k$ to get
$C_k(\boldsymbol{x}_{>k}),U_k,Y_k(x_k)$, form $G_k$ and $F_{k+1}$ per `opts.tucker_update`
(below), then extend the prefix table
$\mathrm{prefixes}_k=\mathrm{prefixes}_{k-1}\times\{x_1^*,\ldots,x_r^*\}$. After stage $d$,
$f$ is evaluated on the resulting $r^d$-point grid to get the core $\mathcal{G}$.

$C_k$ itself is never materialized as an array: because
$[C_k(\boldsymbol{y})]_{\boldsymbol{i},j}=F(\boldsymbol{i};x_j;\boldsymbol{y})$ is literally $f$
evaluated at prefix $\boldsymbol{i}$ extended by pivot $x_j$, the entire next stage is
realized "for free" just by growing the prefix table and calling `f` again.

### Stage update modes (`opts.tucker_update`)

Both modes are exact in infinite precision; they differ in how $U_k$ is distributed.

| Mode | Factor $G_k(x_k)$ | Next stage $F_{k+1}$ |
|---|---|---|
| `"C"` (default) | $U_k\,Y_k(x_k)$ | $C_k(\boldsymbol{x}_{>k})$, exact via prefix encoding |
| `"CU"` | $Y_k(x_k)$ | $C_k(\boldsymbol{x}_{>k})\times_k U_k$ |

`"C"` absorbs $U_k$ into the factor immediately, so nothing further needs correcting.
`"CU"` fits only the well-scaled raw $Y_k$ and stashes $U_k^\top$; once all $d$ stages are
done, `build_chebtucker` applies the accumulated corrections to the raw core,
$\mathcal{G}' = \mathcal{G}\times_1 U_1\times_2\cdots\times_d U_d$. `"CU"` exists mainly to
cross-check `"C"`.

Fitting the transformed $U_k Y_k$ directly vs. fitting raw $Y_k$ and applying $U_k$
to the coefficients afterward are mathematically identical (the Chebyshev transform is
linear), and empirically make no difference to interpolation accuracy — the real source
of numerical blow-up when $S$ is ill-conditioned is cancellation in the final
multilinear core contraction once factor magnitudes have been inflated by a
large-entried $U_k$, not the fitting step itself. `stage_max_abs_U` in the diagnostics
is there to help detect this.

**Removed:** an earlier $L^2$-orthogonalization mode (`"qr"`/`"qr_dense"`, factoring
$C_k=Q_k\tilde R_k$) was unnecessary for correctness and has been removed.

### Cross-matrix inversion (`cross.stable_cross_inverse`)

$U_k=S_k^{-1}$ is computed by tiered solver selected by $\mathrm{cond}(S_k)$:

| $\mathrm{cond}(S_k)$ | Method |
|---|---|
| $<10^8$ | dense `solve` |
| $<$ `max_cross_cond`, `cross_solve=="qr"` | QR inverse (rejected if any $\lvert R_{ii}\rvert$ too small) |
| $<$ `max_cross_cond` (else) | truncated pseudoinverse (`svd_rcond`) |
| $\ge$ `max_cross_cond`, `cross_solve=="tikhonov"` | Tikhonov-regularized inverse |
| $\ge$ `max_cross_cond` (else) | rejected — `nan`-filled, `info.accepted=False` |

`cross_solve_backend` (`"auto"`/`"host"`/`"jax"`) picks NumPy-on-host vs. JAX-on-device;
`"auto"` only uses JAX once rank $\ge$ `cross_solve_jax_min_rank` on a GPU/TPU array.
Every call returns `types.CrossSolveInfo`, feeding `diagnostics["stage_cross_cond"]`.

## Installation

```bash
pip install -e .
```

## Quick start

```python
import jax.numpy as jnp
from chebtucker import build_chebtucker, TuckerOptions

def f(x):           # x: (n, d) float64
    return jnp.exp(-jnp.sum(x**2, axis=-1))

opts = TuckerOptions(max_rank=6, tucker_update="C", cheb_tol=1e-10)
approx = build_chebtucker(f, opts, d=5)

print(approx(jnp.zeros(5)))            # single point
print(approx(jnp.zeros((1000, 5))).shape)  # batch: (1000,)
print(approx.diagnostics)
```

## `TuckerOptions` reference

| Field | Default | Meaning |
|---|---|---|
| `domain` | `None` (→ $[-1,1]^d$) | Physical box, shape `(d, 2)` |
| `max_rank` | `8` | Target Tucker rank $r$ |
| `ranks` | `None` | If set, `ranks[0]` overrides `max_rank` (still one shared rank) |
| `factor_representation` | `"scalar_chebfuns"` | `"scalar_chebfuns"` (default: each of the $r$ columns of a factor is its own adaptively-resolved `chebfunjax` chebfun, `ChebfunMatrixCore`; required by `compute_tools.py`) or `"coeff_tensor"` (all $r$ columns share one Chebyshev-Lobatto grid/degree, `ChebyshevCoeffCore`) |
| `tucker_update` | `"C"` | `"C"` or `"CU"` |
| `tol` | `1e-8` | Reserved for adaptive-rank extensions; unused by `build_chebtucker` |
| `cheb_tol` | `1e-8` | Chebyshev tail threshold for fitting each factor |
| `coeff_core_n_values` | `(17,33,65,129,257)` | Lobatto node-count ladder tried per factor |
| `n_starts` | `256` | Candidates sampled per pivot-search round |
| `boundary_bias_fraction` | `0.25` | Fraction of `n_starts` biased toward boundaries |
| `residual_chunk_size` | `100_000` | Caps peak memory in batched residual evaluation |
| `pivot_abs_tol` | `1e-14` | Residual floor for stopping a stage's search (on the sample, not the true residual) |
| `max_cross_cond` | `1e10` | Ceiling on $\mathrm{cond}(S_k)$ before a pivot is rejected |
| `cross_solve` | `"svd"` | Fallback once $\mathrm{cond}(S)\ge 10^8$: `"solve"`/`"qr"`/`"svd"`/`"tikhonov"` |
| `cross_solve_backend` | `"auto"` | `"auto"`/`"host"`/`"jax"` |
| `cross_solve_jax_min_rank` | `128` | Rank threshold for the JAX backend in `"auto"` mode |
| `svd_rcond` | `1e-10` | Relative singular-value cutoff |
| `tikhonov_lambda` | `1e-14` | Regularization when `cross_solve=="tikhonov"` |
| `random_seed` | `0` | Seeds the single PRNG key used for the whole build |

**Note:** `factor_representation` defaults to `"scalar_chebfuns"`, so every factor $G_k$
is by default a row of independently-adaptive `chebfunjax` chebfuns
(`cores.ChebfunMatrixCore`), fit via `build._build_tucker_factor_chebfun` — this is what
`compute_tools.py` (`tucker_integral`, `tucker_marginal_2d`) expects and requires. Pass
`factor_representation="coeff_tensor"` to instead get the shared-degree
`ChebyshevCoeffCore` fit (`cores.build_coeff_tensor_core_from_matrix_function`); that path
does not support `compute_tools.py`.

## Diagnostics

`ChebTucker.diagnostics`: `tucker_update`, `tucker_rank`; `stage_times_sec`,
`stage_pivot_counts` (accepted pivots per stage — can be less than requested, see
above); `stage_cross_cond`, `stage_max_abs_U`; `total_function_value_points`,
`core_function_points` (= $r^d$), `core_build_time_sec`.

`ChebTucker` also exposes Lebesgue-type conditioning diagnostics:
`qhat_matrix(stage)` returns $\hat Q_k[j,\ell]=G_k(x_k^{(\ell)})[j]$ (the factor evaluated
at its own column-pivot x-values); `qhat_inv_inf_norms()` returns
$\lVert \hat Q_k^{-1}\rVert_\infty$ per stage — large values flag poorly conditioned pivots.

## Package structure

```
chebtucker/
├── build.py            — build_chebtucker; factor builders _build_tucker_factor
│                          (coeff_tensor) / _build_tucker_factor_chebfun
│                          (scalar_chebfuns); pivot search _tucker_select_pivots
├── tucker.py            — ChebTucker: evaluation, qhat_matrix / qhat_inv_inf_norms
├── compute_tools.py      — tucker_integral, tucker_marginal_2d (+ plotting):
│                          O(d) 1D operations exploiting the Tucker factors'
│                          own chebfun .sum()/evaluation; requires
│                          scalar_chebfuns-built factors (see note above)
├── options.py            — TuckerOptions
├── residual.py            — join_x_y, build_tucker_cross_matrix,
│                          tucker_batch_residual_eval (chunked)
├── stage_eval.py           — _TuckerStageEval, _tucker_update_prefixes
├── cores.py               — ChebyshevCoeffCore, ChebfunMatrixCore, Lobatto/DCT fitting
├── cross.py                — stable_cross_inverse
├── domains.py              — normalize_domain, default_domain, stage_bounds
├── sampling.py              — sample_stage_points, expand_samples_over_prefix_indices
└── types.py                 — StageContext, CrossSolveInfo, CandidateBatch
```

## Key design notes

**Tucker vs. TT cross matrix.** TT's $M[i,j]=F(\alpha_i,x_i,\eta_j)$ uses the row
pivot's $x$; Tucker's $S[j,k]=F(\boldsymbol{i}_j;x_k;\boldsymbol{y}_j)$ uses the *column*
pivot's $x$, so $Y(x)[k]=F(\boldsymbol{i}_k,x,\boldsymbol{y}_k)$ never depends on the query
discrete index — only $C$ does.

**Cartesian prefix growth.** TT keeps exactly $r$ prefixes at every stage; Tucker's
prefix count grows as $r^k$, so both the per-round pivot-search candidate pool
(`n_starts * r^{k-1}`) and the final core evaluation ($r^d$) scale accordingly. This is
the dominant cost driver at higher $d$ and $r$.

**`"C"` mode needs no chaining machinery.** Because $F_{k+1}=C_k(\boldsymbol{x}_{>k})$ is
realized exactly through the prefix table, every stage evaluator is simply
`_TuckerStageEval(f, stage_ctx)` — there's no analogue of the old
`TuckerQRProjectedFunction`/`TuckerQRDenseProjectedFunction` chaining classes.

## Current problems
We are currently choosing the pivots very poorly. It just samples `n_starts` points and pivots on the argmax of those. I have been working on porting the pivot optimization scheme from chebttcur and it's working but only on functions fully written in Jax, so I am working on generalizing it a bit.

There is some strange thing that seems to be happenning with the default `factor_representation="scalar_cheb"`. When one tries to approximate a difficult function, when you increase the rank to bring down the max-error, once you get to $\approx0.01$, there is some rank value at which the algorithm just stalls (e.g. I have seen it take ~100 seconds at rank 19 and then run indefinitely for rank 20). If one sets `factor_representation="coeff_tensor"` this issue is resolved. Further, with `factor_representation="coeff_tensor"` one can just recover the univariate chebfuns by a simple bit of code. So I think it could make sense just to remove this option, only do the `factor_representation="coeff_tensor"`, and output the chebfun resolved version of this.
