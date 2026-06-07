"""
_validation_data.py
-------------------
Shared universes and small reporting helpers for the backtest-validation
notebook (``AssetAllocation5_TAA.ipynb``).

The perf-parquet **data loaders** live in :mod:`data.perf_loader`; import them
from there.  This module only carries the asset universes and the two tiny
reporting helpers (``sharpe`` and ``describe_distribution``).

Universes
~~~~~~~~~
* ``MACRO_ASSETS`` — a 10-asset multi-asset cross-section spanning cash,
  government / credit fixed income, regional equity and commodities.
* ``US_INDUSTRIES`` — the 10 MSCI USA GICS sector indices.
* ``EQUITY_STYLES`` — 9 equity style / factor tilts.
"""

from __future__ import annotations

import pandas as pd

from analytics.timeseries_analyzer import calc_sharpe_ratio_rf0

MACRO_ASSETS: list[str] = [
    "MM_US_USD", "FI_GB_US", "FI_IL_US", "FI_IG_US", "FI_HY_US",
    "EQ_US", "EQ_EA", "EQ_EM", "CO_GO", "CO",
]
US_INDUSTRIES: list[str] = [
    "MXUS0CD", "MXUS0CS", "MXUS0EN", "MXUS0FN", "MXUS0HC",
    "MXUS0IN", "MXUS0IT", "MXUS0MT", "MXUS0TC", "MXUS0UT",
]
EQUITY_STYLES: list[str] = [
    "EQ_VA", "EQ_GR", "EQ_MO", "EQ_QL", "EQ_MV",
    "EQ_HD", "EQ_SC", "EQ_LC", "EQ_TC",
]


def sharpe(returns: pd.Series) -> float:
    """Annualised, zero-risk-free Sharpe via
    ``analytics.timeseries_analyzer.calc_sharpe_ratio_rf0``.

    The periods-per-year annualisation factor is inferred from the series'
    ``DatetimeIndex`` by the analytics function.
    """
    series = returns.dropna()
    if len(series) < 2:
        return float("nan")
    return float(calc_sharpe_ratio_rf0(series, annualize=True))


def describe_distribution(values: pd.Series, label: str = "Sharpe") -> str:
    """One-line summary of a distribution (mean / std / 5–95% range)."""
    v = values.dropna()
    return (
        f"{label}: mean={v.mean():+.2f}  std={v.std():.2f}  "
        f"min={v.min():+.2f}  5%={v.quantile(0.05):+.2f}  "
        f"95%={v.quantile(0.95):+.2f}  max={v.max():+.2f}  (n={len(v)})"
    )
