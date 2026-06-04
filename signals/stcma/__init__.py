"""signals.stcma — Short-Term Capital Market Assumptions (momentum signals)."""

from signals.stcma.momentum import (
    CrossSectionalMomentum,
    TimeSeriesMomentum,
    cross_sectional_momentum_returns,
    cross_sectional_momentum_weights,
    time_series_momentum_returns,
)

__all__ = [
    "CrossSectionalMomentum",
    "TimeSeriesMomentum",
    "cross_sectional_momentum_returns",
    "cross_sectional_momentum_weights",
    "time_series_momentum_returns",
]
