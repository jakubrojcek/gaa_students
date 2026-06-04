# backtesting

A general-purpose backtest loop. The engine only ever talks to a `Strategy`; the strategy
bundles its own signal/risk/optimisation logic.

## Contents

- `backtest_engine.py` — `Backtester` + `BacktestConfig` (price→returns conversion, weight drift,
  scheduled rebalancing, one-way transaction costs).
- `backtest_result.py` — `BacktestResult`: `nav`, `returns`, `weights_history`, `turnover`,
  `rebalance_dates`.
- `strategy_protocols.py` — the `Strategy` protocol (single method `compute_weights`) plus the
  `Signal` / `RiskModel` / `Optimizer` component protocols and the `Weights` type alias.
- `rebalance_schedule.py` — `make_rebalance_dates` (supports `B, D, W, ME, MS, BME, BMS, QE, QS, YE, YS`).
- `utils.py` — weight drift, turnover, transaction-cost, and buy-and-hold helpers.

A strategy is anything implementing
`compute_weights(data, as_of, current_weights) -> pd.Series`. The engine expects price-level
(`-i`) columns and converts to returns internally.

## Example

```python
import pandas as pd
from backtesting import BacktestConfig, Backtester

class ConstantWeightStrategy:
    def __init__(self, target: pd.Series) -> None:
        self.target = target
    def compute_weights(self, data, as_of, current_weights):
        return self.target

config = BacktestConfig(
    initial_capital=100.0,
    transaction_cost_bps=10.0,
    rebalance_freq="YE",                 # annual rebalance
    initial_weights_from_strategy=True,
)
target = pd.Series({"S&P 500": 0.6, "U.S. Treasuries": 0.4})
result = Backtester(config).run(prices[list(target.index)], ConstantWeightStrategy(target))

result.nav            # NAV path -> feed into TimeseriesAnalyzer
result.turnover       # one-way turnover per rebalance
```

See `AssetAllocation1_BacktestReturns.ipynb` (constant weights) and
`AssetAllocation4_Robust.ipynb` (a risk-parity strategy driven through the same engine).
