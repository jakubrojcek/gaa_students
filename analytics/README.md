# analytics

Performance and risk metrics for a price (or NAV) panel.

## Contents

- `timeseries_analyzer.py`
  - `TimeseriesAnalyzer` — wraps a prices DataFrame, derives returns, and applies a battery of metrics.
  - Standalone metric functions: `calc_cagr`, `calc_volatility`, `calc_arithmetic_mean_return`,
    `calc_sharpe_ratio`, `calc_sharpe_ratio_rf0`, `calc_sortino_ratio`, `calc_max_drawdown`,
    `probabilistic_sharpe_ratio_from_returns`, `deflated_sharpe_ratio_from_returns`,
    `infer_periods_per_year`.

The analyzer derives returns internally (`pct_change`) and annualises by inferring the data
frequency from the `DatetimeIndex`. Pass `risk_free_col` to use a series as a dynamic risk-free
benchmark for Sharpe/Sortino.

## Example

```python
from analytics.timeseries_analyzer import TimeseriesAnalyzer, calc_max_drawdown

analyzer = TimeseriesAnalyzer(prices, risk_free_col="Money market")
metrics = analyzer.apply_standard_functions()          # CAGR, vol, Sharpe, Sortino, drawdown, VaR/CVaR
print(metrics.loc[["calc_cagr", "calc_volatility", "calc_sharpe_ratio"]])

# Or a chosen subset, annualised:
from analytics.timeseries_analyzer import calc_cagr, calc_volatility
summary = analyzer.apply_functions([calc_cagr, calc_volatility], annualize=True)

# Drawdowns straight from a returns frame:
dd = calc_max_drawdown(prices.pct_change().dropna())
```

Used throughout — see `AssetAllocation1_BacktestReturns.ipynb`.
