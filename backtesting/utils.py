"""
utils.py
--------
Shared helpers for the backtesting module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.strategy_protocols import Weights


def normalize_weights(weights: Weights) -> Weights:
    """Scale weights so they sum to 1.0.

    Parameters
    ----------
    weights : Weights
        Raw portfolio weights.

    Returns
    -------
    Weights
        Rescaled weights summing to 1.0.

    Raises
    ------
    ValueError
        If all weights are zero.
    """
    total = weights.sum()
    if np.isclose(total, 0.0):
        raise ValueError("Cannot normalize weights that sum to zero.")
    return weights / total


def calc_turnover(
    current_weights: Weights,
    target_weights: Weights,
) -> float:
    """Compute turnover as the sum of absolute weight changes.

    Parameters
    ----------
    current_weights : Weights
        Weights before rebalancing.
    target_weights : Weights
        Weights after rebalancing.

    Returns
    -------
    float
        Total turnover (always >= 0).
    """
    # Align indices in case the asset universes differ
    current_aligned, target_aligned = current_weights.align(
        target_weights, fill_value=0.0,
    )
    return float(np.abs(target_aligned - current_aligned).sum())


def apply_transaction_cost(
    nav: float,
    turnover: float,
    cost_bps: float,
) -> float:
    """Deduct proportional transaction costs from NAV.

    Parameters
    ----------
    nav : float
        Current NAV before cost deduction.
    turnover : float
        Turnover at this rebalance.
    cost_bps : float
        One-way transaction cost in basis points.

    Returns
    -------
    float
        NAV after cost deduction.
    """
    if cost_bps <= 0.0 or turnover <= 0.0:
        return nav
    return nav * (1.0 - turnover * cost_bps / 10_000)


def buy_and_hold_nav(prices: pd.Series, initial_capital: float = 100.0) -> pd.Series:
    """Rescale a price series into a buy-and-hold NAV.

    Parameters
    ----------
    prices : pd.Series
        Price-level history (``-i`` convention).  The first value anchors
        the NAV at ``initial_capital``.
    initial_capital : float
        Starting capital for the NAV series.

    Returns
    -------
    pd.Series
        NAV indexed by the same dates as ``prices``.
    """
    return prices / float(prices.iloc[0]) * initial_capital


def drift_weights(
    weights: Weights,
    period_returns: pd.Series,
    portfolio_return: float,
    long_short: bool = False,
) -> Weights:
    """Drift weights forward by one period based on realised returns.

    After one period, an asset with higher return increases in relative
    weight.  This function computes the new weights without rebalancing.

    Parameters
    ----------
    weights : Weights
        Weights at the start of the period.
    period_returns : pd.Series
        Per-asset returns realised during the period.
    portfolio_return : float
        Portfolio-level return for the period.
    long_short : bool
        If ``True``, drift the long and short legs independently so that
        each leg's gross exposure is preserved.  In a standard (long-only)
        portfolio the combined formula ``w*(1+r)/(1+r_p)`` is correct, but
        for long/short portfolios it causes the long and short gross
        exposures to bleed into each other through the total portfolio
        return.  Separate leg drift keeps the portfolio structure intact
        between rebalances.

    Returns
    -------
    Weights
        Drifted weights.
    """
    if np.isclose(1.0 + portfolio_return, 0.0):
        return weights  # portfolio wiped out — weights are undefined

    if not long_short:
        return weights * (1.0 + period_returns) / (1.0 + portfolio_return)

    # --- Long/short: drift each leg independently -----------------------
    long_mask = weights >= 0
    short_mask = weights < 0

    drifted = weights.copy()

    # Drift long leg, rescale to preserve gross long exposure
    if long_mask.any():
        w_long = weights[long_mask]
        r_long = period_returns.reindex(w_long.index, fill_value=0.0)
        long_gross = float(w_long.sum())
        raw_long = w_long * (1.0 + r_long)
        raw_long_sum = float(raw_long.sum())
        if long_gross > 0.0 and not np.isclose(raw_long_sum, 0.0):
            drifted.loc[long_mask] = raw_long * (long_gross / raw_long_sum)

    # Drift short leg, rescale to preserve gross short exposure
    if short_mask.any():
        w_short = weights[short_mask]
        r_short = period_returns.reindex(w_short.index, fill_value=0.0)
        short_gross = float(w_short.sum())  # negative
        raw_short = w_short * (1.0 + r_short)
        raw_short_sum = float(raw_short.sum())
        if short_gross < 0.0 and not np.isclose(raw_short_sum, 0.0):
            drifted.loc[short_mask] = raw_short * (short_gross / raw_short_sum)

    return drifted
