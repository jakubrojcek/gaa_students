# signals.stcma

Momentum signals expressed as portfolio construction rules.

## Contents

- `momentum.py`
  - `time_series_momentum_returns(returns, lookback)` — position = sign of the trailing
    `lookback`-period return, applied to the next period (causal, ∈ {−1, 0, +1}).
  - `cross_sectional_momentum_weights(returns, lookback, quantile, long_only=True)` — rank assets
    by trailing return, hold the top quantile with rank-scaled weights summing to 1.
  - `cross_sectional_momentum_returns(...)` — the resulting strategy return series.
  - `TimeSeriesMomentum(lookbacks)` / `CrossSectionalMomentum(lookbacks, quantiles, long_only=True)`
    — parameterised strategy objects with `.param_grid()`, `.strategy_returns(R, **params)`, and
    `.fit(R, dates)` → `.selected_params_`. These plug straight into `signals.ml_pipeline`.

## Example

```python
from signals.stcma import (
    TimeSeriesMomentum, time_series_momentum_returns, cross_sectional_momentum_weights,
)

# single-asset time-series momentum
strat_ret = time_series_momentum_returns(eq_returns, lookback=12)

# long-only cross-sectional momentum weights (top 30%, rank-scaled)
W = cross_sectional_momentum_weights(panel_returns, lookback=12, quantile=0.3, long_only=True)

# as a tunable object for walk-forward / CPCV
ts = TimeSeriesMomentum([3, 6, 9, 12, 18, 24])
trials = {f"lb{p['lookback']}": ts.strategy_returns(eq_returns, **p) for p in ts.param_grid()}
```

See `AssetAllocation5_TAA.ipynb`.
