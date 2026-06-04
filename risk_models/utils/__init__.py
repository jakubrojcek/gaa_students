"""risk_models.utils — shared utilities for risk model analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def map_dates_to_index(
    dates: list[pd.Timestamp],
    index: pd.DatetimeIndex,
) -> dict[pd.Timestamp, pd.Timestamp]:
    """Map requested dates to the nearest prior date in *index*.

    Parameters
    ----------
    dates : list[pd.Timestamp]
        Dates to look up.
    index : pd.DatetimeIndex
        Available observation dates.

    Returns
    -------
    dict[pd.Timestamp, pd.Timestamp]
        Mapping from each requested date to the nearest prior observation.

    Raises
    ------
    ValueError
        If a requested date precedes the first observation.
    """
    mapping: dict[pd.Timestamp, pd.Timestamp] = {}
    for d in dates:
        prior = index[index <= d]
        if prior.empty:
            raise ValueError(f"No data available on or before {d}")
        mapping[d] = prior[-1]
    return mapping


def cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    """Convert a covariance matrix to a correlation matrix.

    Parameters
    ----------
    cov : pd.DataFrame
        N×N covariance matrix.

    Returns
    -------
    pd.DataFrame
        N×N correlation matrix (same index/columns as input).
    """
    std = np.sqrt(np.diag(cov.values))
    corr = cov.values / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


def extract_rolling_corr(
    rolling_cov: dict[pd.Timestamp, pd.DataFrame],
    asset_a: str,
    asset_b: str,
) -> pd.Series:
    """Extract a pairwise correlation time series from a rolling covariance dict.

    Parameters
    ----------
    rolling_cov : dict[pd.Timestamp, pd.DataFrame]
        Output of :func:`~risk_models.covariance.compute_rolling_covariance`.
    asset_a : str
        First asset name (must be an index/column label in each matrix).
    asset_b : str
        Second asset name.

    Returns
    -------
    pd.Series
        Date-indexed correlation series, sorted chronologically.
    """
    records: dict[pd.Timestamp, float] = {}
    for date, cov in rolling_cov.items():
        std_a = np.sqrt(cov.loc[asset_a, asset_a])
        std_b = np.sqrt(cov.loc[asset_b, asset_b])
        if std_a > 0 and std_b > 0:
            records[date] = cov.loc[asset_a, asset_b] / (std_a * std_b)
    return pd.Series(records).sort_index()
