"""
perf_loader.py
--------------
Convenience loaders for the performance Parquet panels bundled with this
teaching repo (``perf_M.parquet`` / ``perf_Q.parquet`` at the repo root) —
simple-return (``-r``) and price-level (``-i``) columns.

These return clean rectangular panels for the course notebooks; the
backtest-validation notebook (``AssetAllocation5_TAA.ipynb``) feeds the ``-i``
price columns to :class:`backtesting.Backtester` and the ``-r`` return columns
to the cross-sectional helpers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# data/perf_loader.py -> parents[1] == repo root, where the parquet panels live.
_DATA_DIR = Path(__file__).resolve().parents[1]
PERF_PATHS: dict[str, Path] = {
    "M": _DATA_DIR / "perf_M.parquet",
    "Q": _DATA_DIR / "perf_Q.parquet",
}
PERIODS_PER_YEAR: dict[str, float] = {"M": 12.0, "Q": 4.0}


def load_returns(
    assets: list[str],
    frequency: str = "M",
    balanced_panel: bool = True,
) -> pd.DataFrame:
    """Load simple-return (``-r``) columns for ``assets`` at the given frequency.

    Parameters
    ----------
    assets : list[str]
        Asset base names (without the ``-r`` suffix).
    frequency : {"M", "Q"}
        Monthly or quarterly perf parquet.
    balanced_panel : bool
        When ``True`` (default) keep only rows where *every* asset is present
        (a rectangular panel) — the clean input cross-sectional momentum
        expects.  When ``False`` keep rows where at least one asset is present.
    """
    if frequency not in PERF_PATHS:
        raise ValueError(f"frequency must be 'M' or 'Q', got {frequency!r}.")
    df = pd.read_parquet(PERF_PATHS[frequency])
    columns = [f"{a}-r" for a in assets]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing return columns in perf_{frequency}: {missing}")
    out = df[columns].copy()
    out.columns = assets
    return out.dropna(how="any") if balanced_panel else out.dropna(how="all")


def load_single_returns(asset: str = "EQ_US", frequency: str = "M") -> pd.Series:
    """Single-asset simple-return series."""
    return load_returns([asset], frequency=frequency)[asset]


def load_prices(assets: list[str], frequency: str = "M") -> pd.DataFrame:
    """Load price-level (``-i``) columns for a headline ``Backtester`` run."""
    if frequency not in PERF_PATHS:
        raise ValueError(f"frequency must be 'M' or 'Q', got {frequency!r}.")
    df = pd.read_parquet(PERF_PATHS[frequency])
    columns = [f"{a}-i" for a in assets]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing price columns in perf_{frequency}: {missing}")
    return df[columns].dropna(how="any")
