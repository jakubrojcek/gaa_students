"""
signal_strategy.py
------------------
Generic ``Strategy``-protocol implementation that converts a ``Signal``
into portfolio weights via ``w = k * signal``, with an optional delay.

The delay ``delta`` is expressed in **observations** on the price data's
native index.  With ``delta = 0`` the signal is computed on data through
the rebalance date ``t`` and — thanks to the backtester's loop ordering
— the resulting weights apply to the return from ``t`` to ``t + 1``.
With ``delta > 0`` the signal is computed on data through ``t - delta``
instead, introducing an explicit information lag while weights are still
applied starting at ``t + 1``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from backtesting.strategy_protocols import Signal, Weights

LegWeighting = Literal["equally_weighted", "linear", "square_root"]


class SignalStrategy:
    """Translate a per-asset ``Signal`` into weights via ``w = k * signal``.

    Parameters
    ----------
    signal : Signal
        Any object implementing ``generate(data, as_of) -> pd.Series``.
    k : float
        Proportionality constant applied to the raw signal values
        (``weights = k * signal``).  No weight normalisation is performed
        — leverage and partial investment are allowed per the ``Weights``
        contract.
    delta : int
        Non-negative information lag in observations.  ``delta = 0``
        uses data through the rebalance date; ``delta = n`` uses data
        through ``n`` observations earlier.
    """

    def __init__(
        self,
        signal: Signal,
        k: float = 1.0,
        delta: int = 0,
    ) -> None:
        if delta < 0:
            raise ValueError(f"delta must be >= 0, got {delta}.")
        self.signal = signal
        self.k = k
        self.delta = delta

    def compute_weights(
        self,
        data: pd.DataFrame,
        as_of: pd.Timestamp,
        current_weights: Weights | None,
    ) -> Weights:
        """Compute target weights from the signal at ``as_of - delta``.

        Parameters
        ----------
        data : pd.DataFrame
            Price-level history up to and including ``as_of``.
        as_of : pd.Timestamp
            Current rebalance date.
        current_weights : Weights | None
            Ignored by this strategy (kept for protocol compliance).

        Returns
        -------
        Weights
            ``pd.Series`` indexed by asset name.  Assets without enough
            history for the signal receive a zero weight.
        """
        if self.delta == 0:
            effective_data = data
            effective_as_of = as_of
        else:
            as_of_loc = data.index.get_indexer([as_of])[0]
            target_loc = as_of_loc - self.delta
            if target_loc < 0:
                return pd.Series(0.0, index=data.columns)
            effective_data = data.iloc[: target_loc + 1]
            effective_as_of = effective_data.index[-1]

        signal_values = self.signal.generate(effective_data, effective_as_of)
        weights = self.k * signal_values.reindex(data.columns)
        return weights.fillna(0.0)


# ---------------------------------------------------------------------------
# Cross-sectional ranking weights
# ---------------------------------------------------------------------------


def _leg_weights(k: int, leg_weighting: LegWeighting) -> np.ndarray:
    """Within-leg weights for ``k`` names ranked best→worst, summing to 1.

    ``equally_weighted`` gives ``1/k`` to every name; ``linear`` and
    ``square_root`` taper by **rank position** so the best-ranked name carries
    the largest share — the raw units are ``[k, k-1, …, 1]`` (and the
    element-wise square root of those), normalised to sum to 1.
    """
    if leg_weighting == "equally_weighted":
        units = np.ones(k)
    elif leg_weighting == "linear":
        units = np.arange(k, 0, -1, dtype=float)
    elif leg_weighting == "square_root":
        units = np.sqrt(np.arange(k, 0, -1, dtype=float))
    else:
        raise ValueError(
            "leg_weighting must be 'equally_weighted', 'linear' or "
            f"'square_root', got {leg_weighting!r}."
        )
    return units / units.sum()


def cross_sectional_weights(
    trailing_row: pd.Series,
    columns: pd.Index,
    quantile: float,
    long_only: bool = False,
    leg_weighting: LegWeighting = "equally_weighted",
) -> pd.Series:
    """Cross-sectional ranking weights for one period.

    Assets are ranked by ``trailing_row`` (a trailing return or any other
    cross-sectional signal); the top ``quantile`` fraction forms the long leg
    and the bottom ``quantile`` the short leg.  Each leg is weighted within
    itself by ``leg_weighting`` (equal, linear-by-rank or square-root-by-rank,
    best-ranked name heaviest) and normalised so the long leg sums to ``+1`` and
    the short leg to ``-1``.

    With ``long_only=False`` (default) this dollar-neutral long/short overlay
    *is* the portfolio.  With ``long_only=True`` the overlay is added to a
    fully-invested 1/n book (``1/n`` on every rankable asset) and scaled by the
    largest factor that keeps every weight non-negative, so the result is a
    genuinely long-only, fully-invested portfolio (sum ``+1``) whose
    worst-ranked short name lands at exactly zero.

    Returns an all-``NaN`` row when a portfolio cannot be formed (fewer than two
    rankable assets) so the period is excluded rather than counted as a flat
    (zero) return.
    """
    valid = trailing_row.dropna()
    n = len(valid)
    if n < 2:
        return pd.Series(np.nan, index=columns)

    k = min(max(1, int(np.floor(quantile * n))), n // 2)  # disjoint long/short legs

    leg = _leg_weights(k, leg_weighting)
    overlay = pd.Series(0.0, index=columns)
    overlay.loc[valid.nlargest(k).index] = leg
    overlay.loc[valid.nsmallest(k).index] = -leg
    if not long_only:
        return overlay

    # Add the dollar-neutral overlay to a fully-invested 1/n book, scaled by the
    # largest factor that keeps every weight non-negative (the most-shorted name
    # lands at exactly zero) so the result is genuinely long-only and sums to 1.
    base = 1.0 / n
    max_short_weight = float(-overlay.min())
    alpha = base / max_short_weight if max_short_weight > 0.0 else 1.0
    weights = pd.Series(0.0, index=columns)
    weights.loc[valid.index] = base
    return weights + alpha * overlay
