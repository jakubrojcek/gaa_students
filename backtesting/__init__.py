"""
backtesting
-----------
General-purpose backtesting engine for asset allocation strategies.

Public API
~~~~~~~~~~
- ``Backtester`` / ``BacktestConfig``  — engine + configuration
- ``BacktestResult``                   — immutable output container
- ``Strategy``                         — primary protocol (the only one the
  backtester interacts with)
- ``Signal``, ``RiskModel``, ``Optimizer`` — component protocols for
  building strategies
- ``Weights``                          — type alias (``pd.Series``)
- ``make_rebalance_dates``             — rebalance schedule helper
"""

from backtesting.backtest_engine import Backtester, BacktestConfig
from backtesting.backtest_result import BacktestResult
from backtesting.rebalance_schedule import make_rebalance_dates
from backtesting.strategy_protocols import (
    Optimizer,
    RiskModel,
    Signal,
    Strategy,
    Weights,
)
from backtesting.utils import buy_and_hold_nav

__all__ = [
    "Backtester",
    "BacktestConfig",
    "BacktestResult",
    "buy_and_hold_nav",
    "make_rebalance_dates",
    "Optimizer",
    "RiskModel",
    "Signal",
    "Strategy",
    "Weights",
]
