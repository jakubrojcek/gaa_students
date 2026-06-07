"""signals.stcma — momentum signals (time-series & cross-sectional).

The two momentum strategies implement the backtester ``Strategy`` protocol and
are pinned to a single parameter set each; parameter search and walk-forward /
CPCV validation are driven from :mod:`signals.ml_pipeline` via a
``ParameterGrid`` and the ``*_factory`` cell builders, so every reported number
flows through :class:`backtesting.Backtester`.
"""

from signals.stcma.momentum import (
    CrossSectionalMomentum,
    TimeSeriesMomentum,
    cross_sectional_momentum_factory,
    time_series_momentum_factory,
    time_series_momentum_returns,
)
from signals.stcma.signal_strategy import (
    LegWeighting,
    SignalStrategy,
    cross_sectional_weights,
)

__all__ = [
    "CrossSectionalMomentum",
    "LegWeighting",
    "SignalStrategy",
    "TimeSeriesMomentum",
    "cross_sectional_momentum_factory",
    "cross_sectional_weights",
    "time_series_momentum_factory",
    "time_series_momentum_returns",
]
