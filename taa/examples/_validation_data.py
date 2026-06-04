"""
_validation_data.py
-------------------
Shared data loading, universes, and small reporting helpers for the
backtest-validation notebook (``AssetAllocation5_TAA.ipynb``).

Universes
~~~~~~~~~
* ``MACRO_ASSETS`` — a 10-asset multi-asset cross-section spanning cash,
  government / credit fixed income, regional equity and commodities.
* ``US_INDUSTRIES`` — the 10 MSCI USA GICS sector indices.
* ``EQUITY_STYLES`` — 9 equity style / factor tilts.

All series are loaded as **simple returns** (the ``-r`` columns of the perf
parquet) so the momentum strategies operate directly on returns.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analytics.timeseries_analyzer import calc_sharpe_ratio_rf0

# Repo root: taa/examples/_validation_data.py -> parents[2] == repo root,
# where perf_M.parquet lives.
_DATA_DIR = Path(__file__).resolve().parents[2]
PERF_PATHS: dict[str, Path] = {
    "M": _DATA_DIR / "perf_M.parquet",
}
PERIODS_PER_YEAR: dict[str, float] = {"M": 12.0}

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


def load_returns(
    assets: list[str],
    frequency: str = "M",
    balanced: bool = True,
) -> pd.DataFrame:
    """Load simple-return (`-r`) columns for ``assets`` at the given frequency.

    Parameters
    ----------
    assets : list[str]
        Asset base names (without the ``-r`` suffix).
    frequency : {"M"}
        Monthly perf parquet.
    balanced : bool
        When ``True`` (default) keep only rows where *every* asset is present
        (a rectangular panel) — the clean input cross-sectional momentum
        expects.  When ``False`` keep rows where at least one asset is present.
    """
    if frequency not in PERF_PATHS:
        raise ValueError(f"frequency must be 'M', got {frequency!r}.")
    df = pd.read_parquet(PERF_PATHS[frequency])
    columns = [f"{a}-r" for a in assets]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing return columns in perf_{frequency}: {missing}")
    out = df[columns].copy()
    out.columns = assets
    return out.dropna(how="any") if balanced else out.dropna(how="all")


def load_single_returns(asset: str = "EQ_US", frequency: str = "M") -> pd.Series:
    """Single-asset simple-return series."""
    return load_returns([asset], frequency=frequency)[asset]


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
