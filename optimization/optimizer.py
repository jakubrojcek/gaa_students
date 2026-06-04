"""
optimizer.py
------------
Abstract base class for portfolio optimizers.

Provides shared infrastructure — constraint building, penalty
computation, and input validation — that concrete subclasses use.

Planned subclasses:

- ``MeanRiskOptimizer``       — formula-based (mu + Σ): min variance,
  max Sharpe, max utility, target return.
- ``MeanEmpiricalRiskOptimizer`` — scenario-based (mu + returns matrix):
  min CVaR, min CDaR.
- ``RiskBudgetOptimizer``     — risk parity / target risk budgets
  (expected returns optional).

Each subclass supports two solver backends (``"scipy"`` and ``"cvxpy"``)
for redundancy and cross-validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

import cvxpy as cp  # noqa: F401 — used in penalty helpers for cvxpy expressions
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

SolverBackend = Literal["scipy", "cvxpy"]
"""Supported solver backends."""

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseOptimizer(ABC):
    """Abstract base for portfolio optimizers.

    Stores configuration that applies to all optimisation problems and
    provides helper methods for penalty computation and input validation.
    Concrete subclasses define their own ``optimize()`` signatures
    tailored to their inputs.

    Parameters
    ----------
    solver : SolverBackend
        ``"scipy"`` (SLSQP, no extra dependency) or ``"cvxpy"``
        (convex optimisation, richer constraint set).
    long_only : bool
        When ``True`` (default) weights are constrained to be ≥ 0.
    transaction_cost_bps : float
        Proportional transaction cost in basis points.  When > 0 a
        turnover penalty ``tc / 10_000 · Σ|w − w_current|`` is added
        to the objective.  Requires *current_weights* at optimise time.
    tracking_error_aversion : float
        Penalty coefficient for tracking error relative to the
        benchmark.  When > 0 the quadratic term
        ``λ_TE · (w − b)ᵀ Σ (w − b)`` is added to the objective.
        Requires *benchmark* and a covariance matrix at optimise time.
    """

    def __init__(
        self,
        solver: SolverBackend = "scipy",
        long_only: bool = True,
        transaction_cost_bps: float = 0.0,
        tracking_error_aversion: float = 0.0,
    ) -> None:
        self.solver = solver
        self.long_only = long_only
        self.transaction_cost_bps = transaction_cost_bps
        self.tracking_error_aversion = tracking_error_aversion

    # ------------------------------------------------------------------
    # Public interface (subclass-specific signatures)
    # ------------------------------------------------------------------

    @abstractmethod
    def optimize(self, *args, **kwargs) -> pd.Series:
        """Compute optimal portfolio weights.

        Each subclass defines its own signature.  All accept the
        following optional keyword arguments:

        - ``benchmark`` (*pd.Series | None*) — benchmark weights for
          tracking-error penalisation.
        - ``current_weights`` (*pd.Series | None*) — current portfolio
          weights for transaction-cost penalisation.
        """
        ...

    # ------------------------------------------------------------------
    # Penalty helpers (dispatch: numpy arrays → float, cvxpy vars → Expression)
    # ------------------------------------------------------------------

    def _turnover_penalty(
        self,
        weights: np.ndarray,
        current_weights: np.ndarray | None,
    ):
        """L1 turnover cost: ``tc_bps / 10_000 · Σ|w − w_current|``.

        Returns 0 when *current_weights* is ``None`` or
        ``transaction_cost_bps == 0``.

        Works with both numpy arrays (→ float) and cvxpy variables
        (→ ``cp.Expression``).
        """
        if current_weights is None or self.transaction_cost_bps == 0:
            return 0.0
        tc = self.transaction_cost_bps / 10_000
        if isinstance(weights, np.ndarray):
            return tc * float(np.sum(np.abs(weights - current_weights)))
        return tc * cp.norm1(weights - current_weights)

    def _tracking_error_penalty(
        self,
        weights: np.ndarray,
        benchmark: np.ndarray | None,
        cov: np.ndarray,
    ):
        """Quadratic TE penalty: ``λ_TE · (w − b)ᵀ Σ (w − b)``.

        Returns 0 when *benchmark* is ``None`` or
        ``tracking_error_aversion == 0``.

        Works with both numpy arrays (→ float) and cvxpy variables
        (→ ``cp.Expression``).
        """
        if benchmark is None or self.tracking_error_aversion == 0:
            return 0.0
        lam = self.tracking_error_aversion
        if isinstance(weights, np.ndarray):
            diff = weights - benchmark
            return lam * float(diff @ cov @ diff)
        return lam * cp.quad_form(weights - benchmark, cp.psd_wrap(cov))

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_covariance(self, covariance: pd.DataFrame) -> None:
        """Check covariance is square, symmetric, and PSD."""
        n = len(covariance)

        if covariance.shape != (n, n):
            raise ValueError(f"Covariance must be square, got {covariance.shape}")

        vals = covariance.values
        if not np.allclose(vals, vals.T, atol=1e-10):
            raise ValueError("Covariance matrix is not symmetric")

        eigvals = np.linalg.eigvalsh(vals)
        if eigvals.min() < -1e-8:
            raise ValueError(
                f"Covariance matrix is not positive semi-definite "
                f"(min eigenvalue = {eigvals.min():.2e})"
            )

    def _validate_expected_returns(
        self,
        expected_returns: pd.Series,
        assets: Sequence[str],
    ) -> None:
        """Check expected returns cover all assets in the covariance."""
        missing = set(assets) - set(expected_returns.index)
        if missing:
            raise ValueError(
                f"Expected returns missing assets present in covariance: {missing}"
            )

    def _align_weights(
        self,
        weights: pd.Series | None,
        assets: Sequence[str],
        label: str,
    ) -> np.ndarray | None:
        """Reindex optional weight vector to match asset order.

        Returns ``None`` if *weights* is ``None``, otherwise a numpy
        array aligned to *assets* (missing entries filled with 0).
        """
        if weights is None:
            return None
        try:
            aligned = weights.reindex(assets, fill_value=0.0)
        except Exception as exc:
            raise ValueError(f"Cannot align {label} to asset universe: {exc}") from exc
        return aligned.values
