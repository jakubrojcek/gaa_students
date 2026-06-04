"""
strategy_protocols.py
---------------------
Protocol definitions for the backtesting framework.

The primary protocol is ``Strategy`` — the only interface the ``Backtester``
interacts with.  Additional protocols (``Signal``, ``RiskModel``, ``Optimizer``)
are provided for strategy authors who want to compose reusable building blocks
inside their ``Strategy.compute_weights()`` implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Weights = pd.Series
"""Portfolio weights keyed by asset name.

Not constrained to sum to 1.0 — strategies may express leverage,
partial investment, or explicit cash positions.
"""

# ---------------------------------------------------------------------------
# Primary protocol — used by the Backtester
# ---------------------------------------------------------------------------


@runtime_checkable
class Strategy(Protocol):
    """Compute target portfolio weights at a point in time.

    This is the **only** protocol the ``Backtester`` calls.  A strategy
    bundles whatever internal logic it needs (signals, risk models,
    optimizers) and exposes a single ``compute_weights`` method.
    """

    def compute_weights(
        self,
        data: pd.DataFrame,
        as_of: pd.Timestamp,
        current_weights: Weights | None,
    ) -> Weights:
        """Return target weights for the next holding period.

        Parameters
        ----------
        data : pd.DataFrame
            Full price history available up to and including *as_of*.
            The strategy must **not** look beyond *as_of*.
        as_of : pd.Timestamp
            The current rebalancing date.
        current_weights : Weights | None
            Portfolio weights just before rebalancing (after drift).
            ``None`` on the very first call (no prior portfolio).

        Returns
        -------
        Weights
            Target portfolio weights (``pd.Series`` indexed by asset name).
        """
        ...


# ---------------------------------------------------------------------------
# Component protocols — for use *inside* strategy implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class Signal(Protocol):
    """Produce a per-asset score or forecast at a point in time."""

    def generate(
        self,
        data: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.Series:
        """Return a Series of signal values keyed by asset name.

        Parameters
        ----------
        data : pd.DataFrame
            History up to and including *as_of*.
        as_of : pd.Timestamp
            Reference date (no look-ahead allowed).
        """
        ...


@runtime_checkable
class RiskModel(Protocol):
    """Estimate a covariance matrix (or other risk structure)."""

    def estimate(
        self,
        data: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return an N×N covariance matrix as a DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            History up to and including *as_of*.
        as_of : pd.Timestamp
            Reference date.
        """
        ...


@runtime_checkable
class Optimizer(Protocol):
    """Compute target weights given signals and a risk estimate."""

    def optimize(
        self,
        signals: pd.Series,
        risk_model: pd.DataFrame,
        current_weights: Weights,
        constraints: dict | None = None,
    ) -> Weights:
        """Return target portfolio weights.

        Parameters
        ----------
        signals : pd.Series
            Per-asset signal values.
        risk_model : pd.DataFrame
            Covariance matrix (N×N).
        current_weights : Weights
            Current portfolio weights.
        constraints : dict | None
            Optional strategy-specific constraints.
        """
        ...
