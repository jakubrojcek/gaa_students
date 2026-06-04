"""
rebalance_schedule.py
---------------------
Helpers for determining which dates in a time series index are rebalancing
dates according to a given frequency.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

RebalanceFreq = Literal[
    "B",    # Business day
    "D",    # Calendar day
    "W",    # Weekly
    "ME",   # Month end
    "MS",   # Month start
    "BME",  # Business month end
    "BMS",  # Business month start
    "QE",   # Quarter end
    "QS",   # Quarter start
    "YE",   # Year end
    "YS",   # Year start
]


def make_rebalance_dates(
    index: pd.DatetimeIndex,
    freq: RebalanceFreq | str,
) -> list[pd.Timestamp]:
    """Return the subset of *index* dates that fall on rebalancing boundaries.

    The function groups the index by *freq* periods and picks the **last**
    date in each group that actually appears in the index.  This ensures
    every returned date exists in the original data.

    Parameters
    ----------
    index : pd.DatetimeIndex
        The full date index of the returns / prices DataFrame.
    freq : RebalanceFreq | str
        A pandas offset alias (e.g. ``"ME"``, ``"QE"``, ``"YE"``).

    Returns
    -------
    list[pd.Timestamp]
        Sorted list of rebalancing dates drawn from *index*.

    Examples
    --------
    >>> idx = pd.bdate_range("2020-01-01", periods=252, freq="B")
    >>> reb = make_rebalance_dates(idx, "ME")
    >>> len(reb)  # roughly 12 month-end dates
    12
    """
    if len(index) < 2:
        return list(index)

    # Group by the target period and take the last available date per group.
    grouper = index.to_series().groupby(pd.Grouper(freq=freq))
    rebalance_dates = grouper.last().dropna().tolist()

    return sorted(set(rebalance_dates))
