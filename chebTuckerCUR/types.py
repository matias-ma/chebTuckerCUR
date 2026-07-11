"""Shared datatypes for Tucker-qTensCUR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from .domains import normalize_domain


@dataclass(frozen=True)
class CrossSolveInfo:
    """Diagnostics for one cross-matrix inverse computation."""

    accepted: bool
    method: str
    cond: float
    sigma_min: float
    sigma_max: float
    effective_rank: int | None = None
    truncated_rank: int | None = None


@dataclass(frozen=True)
class CandidateBatch:
    """Pivot candidates for one Tucker stage.

    Each row corresponds to one candidate triple ``(alpha, x, y)`` after
    sampling.  ``z_final`` stores the full remaining variable

        z = (x_k, ..., x_d).
    """

    alpha_ids: jnp.ndarray
    z_final: jnp.ndarray
    residual_values: jnp.ndarray
    loss_values: jnp.ndarray
    base_values: jnp.ndarray
    optimizer_value_function_points: int = 0
    optimizer_gradient_points: int = 0
    optimizer_gradient_function_points_estimate: int = 0

    def __post_init__(self) -> None:
        alpha_ids = jnp.asarray(self.alpha_ids, dtype=jnp.int32)
        z_final = jnp.asarray(self.z_final, dtype=jnp.float64)
        residual_values = jnp.asarray(self.residual_values, dtype=jnp.float64)
        loss_values = jnp.asarray(self.loss_values, dtype=jnp.float64)
        base_values = jnp.asarray(self.base_values, dtype=jnp.float64)

        n = alpha_ids.shape[0]
        if z_final.ndim != 2 or z_final.shape[0] != n:
            raise ValueError(f"z_final must have shape ({n}, m), got {z_final.shape}")
        if residual_values.shape != (n,):
            raise ValueError(f"residual_values must have shape ({n},), got {residual_values.shape}")
        if loss_values.shape != (n,):
            raise ValueError(f"loss_values must have shape ({n},), got {loss_values.shape}")
        if base_values.shape != (n,):
            raise ValueError(f"base_values must have shape ({n},), got {base_values.shape}")

        object.__setattr__(self, "alpha_ids", alpha_ids)
        object.__setattr__(self, "z_final", z_final)
        object.__setattr__(self, "residual_values", residual_values)
        object.__setattr__(self, "loss_values", loss_values)
        object.__setattr__(self, "base_values", base_values)
        object.__setattr__(self, "optimizer_value_function_points", int(self.optimizer_value_function_points))
        object.__setattr__(self, "optimizer_gradient_points", int(self.optimizer_gradient_points))
        object.__setattr__(
            self,
            "optimizer_gradient_function_points_estimate",
            int(self.optimizer_gradient_function_points_estimate),
        )


@dataclass(frozen=True)
class StageContext:
    """Context needed to evaluate a Tucker stage from prefixes.

    At Tucker stage ``k``:
    - ``prefixes`` has shape ``(r^{k-1}, k-1)`` — all Cartesian-product
      combinations of the first ``k-1`` pivot x-values.
    - ``rem_dim = d - k + 1`` is the dimension of ``z_rem = (x_k, ..., x_d)``.

    The stage evaluator then computes

        F_k(alpha, z_rem) = f(prefixes[alpha], z_rem).
    """

    stage: int
    d: int
    prefixes: jnp.ndarray
    domain: jnp.ndarray
    rem_dim: int = field(init=False)

    def __post_init__(self) -> None:
        if self.stage < 1 or self.stage > self.d:
            raise ValueError(f"stage must be in [1, {self.d}], got {self.stage}")

        prefixes = jnp.asarray(self.prefixes, dtype=jnp.float64)
        if prefixes.ndim != 2:
            raise ValueError(f"prefixes must be 2D, got shape {prefixes.shape}")
        if prefixes.shape[1] != self.stage - 1:
            raise ValueError(
                f"prefixes second dimension must be {self.stage - 1}, got {prefixes.shape[1]}"
            )

        domain = normalize_domain(self.domain, d=self.d)
        object.__setattr__(self, "prefixes", prefixes)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "rem_dim", self.d - self.stage + 1)
