# signals.stcma

Momentum signals expressed as backtester `Strategy` objects, so every reported
number flows through `backtesting.Backtester`.

## Contents

- `momentum.py`
  - `time_series_momentum_returns(returns, lookback)` — pure signal-to-returns
    map: position = sign of the trailing `lookback`-period return, applied to the
    next period (causal, ∈ {−1, 0, +1}). Kept as a diagnostic benchmark.
  - `TimeSeriesMomentum(lookback)` — single-asset trend follower implementing the
    `Strategy` protocol (`compute_weights`); with `BacktestConfig(long_short=True)`
    it is a fully long / flat / short single-asset book.
  - `CrossSectionalMomentum(lookback, quantile, long_only=False, leg_weighting=...)`
    — ranks assets by trailing return and forms the long/short (or, with
    `long_only=True`, fully-invested long-only) momentum portfolio.
  - `time_series_momentum_factory` / `cross_sectional_momentum_factory` — build a
    concrete strategy from one `ParameterGrid` cell; handed to the validation
    helpers in `signals.ml_pipeline`.
- `signal_strategy.py` — `cross_sectional_weights` (the ranking → weights map),
  the `LegWeighting` options, and `SignalStrategy` (a generic `Signal` → `Strategy`
  adapter).

## Example

```python
from backtesting import BacktestConfig, Backtester
from signals.stcma import TimeSeriesMomentum, CrossSectionalMomentum

# single-asset trend follower as an engine Strategy
bt = Backtester(BacktestConfig(long_short=True)).run(
    prices=eq_prices, strategy=TimeSeriesMomentum(lookback=12)
)

# long-only cross-sectional momentum (top 30%, equal-weighted legs)
xs = CrossSectionalMomentum(lookback=12, quantile=0.3, long_only=True)
weights = xs.compute_weights(panel_prices, panel_prices.index[-1], None)
```

See `AssetAllocation5_TAA.ipynb`.
