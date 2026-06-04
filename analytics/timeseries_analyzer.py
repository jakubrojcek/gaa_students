"""
timeseries_analyzer.py
----------------------
Evaluate the performance of financial strategies from price series.

Dependencies: pandas, numpy, xlwings
"""

import inspect
import math
import os
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Frequency inference
# ---------------------------------------------------------------------------

# pandas freq alias → annualisation factor
_FREQ_MAP: dict[str, float] = {
    # Daily
    "D": 365.0,
    "B": 252.0,
    # Weekly
    "W": 52.0,
    "W-MON": 52.0,
    "W-TUE": 52.0,
    "W-WED": 52.0,
    "W-THU": 52.0,
    "W-FRI": 52.0,
    "W-SAT": 52.0,
    "W-SUN": 52.0,
    # Monthly
    "M": 12.0,
    "ME": 12.0,
    "MS": 12.0,
    "BM": 12.0,
    "BMS": 12.0,
    "CBM": 12.0,
    "CBMS": 12.0,
    # Quarterly
    "Q": 4.0,
    "QE": 4.0,
    "QS": 4.0,
    "BQ": 4.0,
    "BQS": 4.0,
    # Annual
    "A": 1.0,
    "AS": 1.0,
    "BA": 1.0,
    "BAS": 1.0,
    "Y": 1.0,
    "YE": 1.0,
    "YS": 1.0,
    # Sub-daily
    "H": 252.0 * 6.5,
    "T": 252.0 * 6.5 * 60,
    "min": 252.0 * 6.5 * 60,
    "S": 252.0 * 6.5 * 3600,
}


def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    """
    Infer the number of observations per year from a DatetimeIndex.

    Strategy
    --------
    1. Try to read the freq / inferred_freq attribute and look it up in
       ``_FREQ_MAP``.
    2. Fall back to computing the average inter-observation gap in calendar
       days and dividing 365 by it.
    """
    if len(index) < 2:
        raise ValueError(
            "Cannot infer frequency from an index with fewer than 2 observations."
        )

    freq_str: str | None = None
    if hasattr(index, "freq") and index.freq is not None:
        freq_str = index.freq.name
    elif hasattr(index, "inferred_freq") and index.inferred_freq is not None:
        freq_str = index.inferred_freq

    if freq_str is not None:
        # Strip anchoring suffix like "-DEC" from quarterly/annual aliases
        base_alias = freq_str.split("-")[0].upper()
        if base_alias in _FREQ_MAP:
            return _FREQ_MAP[base_alias]
        if freq_str.upper() in _FREQ_MAP:
            return _FREQ_MAP[freq_str.upper()]

    warnings.warn(
        f"Could not resolve frequency alias '{freq_str}'. "
        "Falling back to density-based inference.",
        RuntimeWarning,
        stacklevel=2,
    )
    total_days: float = (index[-1] - index[0]).days
    avg_days_per_obs: float = total_days / (len(index) - 1)
    return 365.0 / avg_days_per_obs


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def _validate_datetime_index(obj: pd.Series | pd.DataFrame) -> None:
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise TypeError(
            f"Expected a DatetimeIndex, got {type(obj.index).__name__}. "
            "Please convert the index before calling this function."
        )


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def calc_cagr(
    returns: pd.Series | pd.DataFrame,
    annualize: bool = True,
) -> float | pd.Series:
    """
    Compound Annual Growth Rate (CAGR).

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Period returns (not prices). Must have a DatetimeIndex.
    annualize : bool
        If True, returns the annualised CAGR.
        If False, returns the total compounded return over the full period.

    Returns
    -------
    float or pd.Series
        CAGR per strategy column (or scalar for a Series input).
    """
    _validate_datetime_index(returns)
    total_return: float | pd.Series = (1 + returns).prod() - 1

    if not annualize:
        return total_return

    n = len(returns)
    years = n / infer_periods_per_year(returns.index)
    if years <= 0:
        raise ValueError("Cannot compute CAGR: effective number of years is <= 0.")

    return (1 + total_return) ** (1.0 / years) - 1


def calc_volatility(
    returns: pd.Series | pd.DataFrame,
    annualize: bool = True,
) -> float | pd.Series:
    """
    Standard deviation of returns (volatility).

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Period returns. Must have a DatetimeIndex.
    annualize : bool
        If True, scales by sqrt(periods_per_year).

    Returns
    -------
    float or pd.Series
    """
    _validate_datetime_index(returns)
    vol: float | pd.Series = returns.std(ddof=1)

    if annualize:
        vol = vol * np.sqrt(infer_periods_per_year(returns.index))

    return vol


def calc_sharpe_ratio(
    returns: pd.Series | pd.DataFrame,
    risk_free_rate: pd.Series,
    annualize: bool = True,
) -> float | pd.Series:
    """
    Sharpe Ratio using a dynamic (time-series) risk-free rate.

    Sharpe = mean(excess_return) / std(excess_return)  [× sqrt(N) if annualised]

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Strategy period returns. Must have a DatetimeIndex.
    risk_free_rate : pd.Series
        Period risk-free returns aligned to the same index.
    annualize : bool

    Returns
    -------
    float or pd.Series
    """
    _validate_datetime_index(returns)
    excess: pd.Series | pd.DataFrame = returns.subtract(risk_free_rate, axis=0)
    sharpe = excess.mean() / excess.std(ddof=1)

    if annualize:
        sharpe = sharpe * np.sqrt(infer_periods_per_year(returns.index))

    return sharpe


def calc_sharpe_ratio_rf0(
    returns: pd.Series | pd.DataFrame,
    annualize: bool = True,
) -> float | pd.Series:
    """
    Sharpe Ratio assuming a **zero risk-free rate** (``rf = 0``).

    Sharpe = mean(return) / std(return)  [× sqrt(N) if annualised]

    Reported alongside :func:`calc_sharpe_ratio` (the proper, risk-free-adjusted
    Sharpe) so the effect of the risk-free assumption is visible at a glance:
    when the two rows are equal, the chosen benchmark risk-free rate was zero
    (or none was supplied) and made no difference; when they diverge, the gap is
    exactly the risk-free drag.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Strategy period returns. Must have a DatetimeIndex.
    annualize : bool

    Returns
    -------
    float or pd.Series
    """
    _validate_datetime_index(returns)
    sharpe = returns.mean() / returns.std(ddof=1)

    if annualize:
        sharpe = sharpe * np.sqrt(infer_periods_per_year(returns.index))

    return sharpe


def calc_sortino_ratio(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series,
    annualize: bool = True,
) -> float | pd.Series:
    """
    Sortino Ratio using a dynamic benchmark for downside deviation.

    Sortino = mean(excess_return) / downside_std(excess_return)  [× sqrt(N) if annualised]

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Strategy period returns. Must have a DatetimeIndex.
    benchmark : pd.Series
        Period benchmark returns used as the Minimum Acceptable Return (MAR).
    annualize : bool

    Returns
    -------
    float or pd.Series
    """
    _validate_datetime_index(returns)
    excess: pd.Series | pd.DataFrame = returns.subtract(benchmark, axis=0)

    def _downside_std(col: pd.Series) -> float:
        downside = col[col < 0]
        if len(downside) == 0:
            return np.nan
        return np.sqrt((downside**2).mean())

    if isinstance(excess, pd.Series):
        downside_dev: float | pd.Series = _downside_std(excess)
        mean_excess: float | pd.Series = excess.mean()
    else:
        downside_dev = excess.apply(_downside_std)
        mean_excess = excess.mean()

    sortino = mean_excess / downside_dev

    if annualize:
        sortino = sortino * np.sqrt(infer_periods_per_year(returns.index))

    return sortino


def calc_max_drawdown(
    returns: pd.Series | pd.DataFrame,
    annualize: bool = False,  # not meaningful for drawdown depth; kept for API consistency
) -> dict[str, Any] | pd.DataFrame:
    """
    Maximum Drawdown analysis.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Strategy period returns. Must have a DatetimeIndex.
    annualize : bool
        Unused for drawdown (included for API consistency).

    Returns
    -------
    dict (for Series input) or pd.DataFrame (for DataFrame input)
        Keys / rows:
            - max_drawdown        : float     — maximum drawdown as a negative decimal
            - peak_date           : Timestamp
            - trough_date         : Timestamp
            - recovery_date       : Timestamp or NaT
            - drawdown_duration   : int       — periods from peak to trough
            - recovery_duration   : int | NaN — periods from trough to recovery
    """
    _validate_datetime_index(returns)

    def _mdd_series(ret: pd.Series) -> dict[str, Any]:
        cum = (1 + ret).cumprod()
        rolling_max = cum.cummax()
        drawdown = cum / rolling_max - 1

        trough_ts = drawdown.idxmin() if not drawdown.empty else drawdown.index[0]
        trough_loc: int = drawdown.index.get_loc(trough_ts)

        peak_ts = rolling_max.iloc[: trough_loc + 1].idxmax()
        peak_loc: int = drawdown.index.get_loc(peak_ts)

        peak_value: float = float(rolling_max.iloc[peak_loc])
        post_trough = cum.iloc[trough_loc:]
        recovered = post_trough[post_trough >= peak_value]

        recovery_date: Any = recovered.index[0] if not recovered.empty else pd.NaT
        recovery_duration: Any = (
            len(post_trough.loc[: recovered.index[0]]) - 1
            if not recovered.empty
            else np.nan
        )

        return {
            "max_drawdown": float(drawdown.loc[trough_ts]),
            "peak_date": peak_ts,
            "trough_date": trough_ts,
            "recovery_date": recovery_date,
            "drawdown_duration": trough_loc - peak_loc,
            "recovery_duration": recovery_duration,
        }

    if isinstance(returns, pd.Series):
        return _mdd_series(returns)

    return pd.DataFrame({col: _mdd_series(returns[col]) for col in returns.columns})


def calc_arithmetic_mean_return(
    returns: pd.Series | pd.DataFrame,
    annualize: bool = True,
) -> float | pd.Series:
    """
    Arithmetic average return.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Period returns. Must have a DatetimeIndex.
    annualize : bool
        If True, scales by periods_per_year.

    Returns
    -------
    float or pd.Series
    """
    _validate_datetime_index(returns)
    mean_ret: float | pd.Series = returns.mean()

    if annualize:
        mean_ret = mean_ret * infer_periods_per_year(returns.index)

    return mean_ret


def _calc_var_single(returns_series: pd.Series, confidence_level: float) -> float:
    """VaR for a single return series (negative value = potential loss)."""
    return float(returns_series.quantile(1.0 - confidence_level))


def _calc_cvar_single(returns_series: pd.Series, confidence_level: float) -> float:
    """CVaR / Expected Shortfall for a single return series."""
    var_threshold = _calc_var_single(returns_series, confidence_level)
    tail = returns_series[returns_series <= var_threshold]
    return float(tail.mean()) if len(tail) > 0 else np.nan


def calc_var_95(
    returns: pd.Series | pd.DataFrame,
    annualize: bool = False,
) -> float | pd.Series:
    """
    Value at Risk at 95% confidence level (5th percentile return, native frequency).

    A negative value represents the potential loss. ``annualize`` is ignored;
    VaR is always reported at the native frequency of the series.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Period returns. Must have a DatetimeIndex.
    annualize : bool
        Unused (kept for API consistency with other metric functions).

    Returns
    -------
    float or pd.Series
    """
    _validate_datetime_index(returns)
    if isinstance(returns, pd.Series):
        return _calc_var_single(returns.dropna(), 0.95)
    return returns.apply(lambda col: _calc_var_single(col.dropna(), 0.95))


def calc_var_99(
    returns: pd.Series | pd.DataFrame,
    annualize: bool = False,
) -> float | pd.Series:
    """
    Value at Risk at 99% confidence level (1st percentile return, native frequency).

    A negative value represents the potential loss. ``annualize`` is ignored;
    VaR is always reported at the native frequency of the series.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Period returns. Must have a DatetimeIndex.
    annualize : bool
        Unused (kept for API consistency with other metric functions).

    Returns
    -------
    float or pd.Series
    """
    _validate_datetime_index(returns)
    if isinstance(returns, pd.Series):
        return _calc_var_single(returns.dropna(), 0.99)
    return returns.apply(lambda col: _calc_var_single(col.dropna(), 0.99))


def calc_cvar_95(
    returns: pd.Series | pd.DataFrame,
    annualize: bool = False,
) -> float | pd.Series:
    """
    Conditional VaR (Expected Shortfall) at 95% confidence level (native frequency).

    Mean return in the worst 5% of observations. ``annualize`` is ignored;
    CVaR is always reported at the native frequency of the series.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Period returns. Must have a DatetimeIndex.
    annualize : bool
        Unused (kept for API consistency with other metric functions).

    Returns
    -------
    float or pd.Series
    """
    _validate_datetime_index(returns)
    if isinstance(returns, pd.Series):
        return _calc_cvar_single(returns.dropna(), 0.95)
    return returns.apply(lambda col: _calc_cvar_single(col.dropna(), 0.95))


def calc_cvar_99(
    returns: pd.Series | pd.DataFrame,
    annualize: bool = False,
) -> float | pd.Series:
    """
    Conditional VaR (Expected Shortfall) at 99% confidence level (native frequency).

    Mean return in the worst 1% of observations. ``annualize`` is ignored;
    CVaR is always reported at the native frequency of the series.

    Parameters
    ----------
    returns : pd.Series or pd.DataFrame
        Period returns. Must have a DatetimeIndex.
    annualize : bool
        Unused (kept for API consistency with other metric functions).

    Returns
    -------
    float or pd.Series
    """
    _validate_datetime_index(returns)
    if isinstance(returns, pd.Series):
        return _calc_cvar_single(returns.dropna(), 0.99)
    return returns.apply(lambda col: _calc_cvar_single(col.dropna(), 0.99))


def calc_rolling_describe(
    returns: pd.Series,
    window_years: float = 1.0,
    annualize: bool = False,
) -> pd.DataFrame:
    """
    Rolling descriptive statistics over a 1-year window at native frequency.

    Parameters
    ----------
    returns : pd.Series
        Period returns with a DatetimeIndex.
    window_years : float
        Window size in years; converted to integer periods using inferred frequency.
    annualize : bool
        Unused (kept for API consistency).

    Returns
    -------
    pd.DataFrame
        Columns: count, mean, std, min, 25%, 50%, 75%, max.
        Index: same DatetimeIndex as input.
    """
    _validate_datetime_index(returns)
    periods_per_year = infer_periods_per_year(returns.index)
    window = max(2, int(round(window_years * periods_per_year)))

    rolled = returns.rolling(window=window, min_periods=1)
    return pd.DataFrame(
        {
            "count": rolled.count(),
            "mean": rolled.mean(),
            "std": rolled.std(ddof=1),
            "min": rolled.min(),
            "25%": rolled.quantile(0.25),
            "50%": rolled.quantile(0.50),
            "75%": rolled.quantile(0.75),
            "max": rolled.max(),
        },
        index=returns.index,
    )


# ---------------------------------------------------------------------------
# Normal distribution helpers (self-contained — no scipy dependency)
# ---------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    """Standard-normal cumulative distribution function via ``math.erf``."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard-normal quantile (inverse CDF), Acklam's rational approximation.

    Accurate to ~1.15e-9 over ``(0, 1)``; a single Halley refinement step
    using :func:`_norm_cdf` polishes the result to full double precision.
    Implemented locally to avoid pulling in ``scipy`` for two tail
    evaluations.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in the open interval (0, 1), got {p}.")

    a = (
        -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
        1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0,
    )
    b = (
        -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
        6.680131188771972e1, -1.328068155288572e1,
    )
    c = (
        -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
        -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0,
    )
    d = (
        7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
        3.754408661907416e0,
    )
    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )

    # One Halley step: refine x so that _norm_cdf(x) == p to machine precision.
    err = _norm_cdf(x) - p
    pdf = math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)
    x = x - err / pdf / (1.0 + x * err / pdf / 2.0)
    return float(x)


# ---------------------------------------------------------------------------
# Sharpe-ratio significance under multiple testing (Bailey & López de Prado)
# ---------------------------------------------------------------------------


def probabilistic_sharpe_ratio(
    sr: float,
    sr_benchmark: float,
    n: int,
    skew: float,
    kurt: float,
) -> float:
    r"""Probabilistic Sharpe Ratio — Bailey & López de Prado (2012).

    Probability that the *true* Sharpe exceeds a benchmark Sharpe
    ``sr_benchmark``, given an observed Sharpe ``sr`` estimated from ``n``
    returns whose higher moments are ``skew`` and ``kurt``:

    .. math::

        \mathrm{PSR}(SR^*) = \Phi\!\left(
            \frac{(\hat{SR} - SR^*)\,\sqrt{n - 1}}
                 {\sqrt{1 - \gamma_3\,\hat{SR} + \frac{\gamma_4 - 1}{4}\,\hat{SR}^2}}
        \right)

    The denominator is the Mertens (2002) non-normal correction to the
    standard error of the Sharpe estimator: negative skew and/or fat tails
    inflate it, deflating the probability.

    Parameters
    ----------
    sr : float
        Observed Sharpe :math:`\hat{SR}`, **per observation** (NOT
        annualised), in the same units as ``sr_benchmark`` and consistent
        with the per-period ``skew`` / ``kurt``.
    sr_benchmark : float
        Benchmark Sharpe :math:`SR^*` to beat (same per-period units).
        ``0.0`` tests whether the true Sharpe is positive.
    n : int
        Number of return observations behind ``sr``.
    skew : float
        Sample skewness :math:`\gamma_3` of the returns.
    kurt : float
        Sample kurtosis :math:`\gamma_4`, **non-excess** (Gaussian → 3, so a
        normal series contributes :math:`(3-1)/4 = 0.5\,\hat{SR}^2`).

    Returns
    -------
    float
        PSR ∈ (0, 1).
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}.")
    variance_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if variance_term <= 0.0:
        raise ValueError(
            "Non-positive variance term in PSR denominator "
            f"(1 - skew·sr + (kurt-1)/4·sr² = {variance_term:.4f}); check that "
            "'kurt' is non-excess and that the moments are mutually consistent."
        )
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(variance_term)
    return _norm_cdf(z)


# Euler–Mascheroni constant, used in the Gumbel expected-maximum term.
_EULER_MASCHERONI = 0.5772156649015329


def deflated_sharpe_ratio(
    sr: float,
    trial_sharpes: Sequence[float] | np.ndarray,
    n: int,
    skew: float,
    kurt: float,
    n_effective_trials: int | None = None,
) -> float:
    r"""Deflated Sharpe Ratio — Bailey & López de Prado (2014).

    The DSR is the PSR of the *selected* strategy benchmarked against the
    **expected maximum** Sharpe attainable by chance after running ``N``
    independent trials whose true Sharpe is zero:

    .. math::

        SR^* = \sqrt{\mathrm{Var}[\{SR_m\}]}\,\Bigl[
            (1 - \gamma)\,Z^{-1}\!\bigl(1 - \tfrac{1}{N}\bigr)
            + \gamma\,Z^{-1}\!\bigl(1 - \tfrac{1}{N\,e}\bigr)
        \Bigr]

    with :math:`\gamma` the Euler–Mascheroni constant, :math:`Z^{-1}` the
    standard-normal quantile, and :math:`\mathrm{Var}[\{SR_m\}]` the
    cross-sectional variance of the ``N`` trial Sharpe estimates.  The
    bracketed factor is the Gumbel extreme-value approximation to the
    expected maximum of ``N`` i.i.d. standard normals.  ``DSR = PSR(SR*)``.

    Effective-N caveat
    ------------------
    The Gumbel term assumes the trials are **independent**.  Parameter
    sweeps are highly correlated (lookbacks 11 and 12 are nearly the same
    strategy), so the *raw* grid size overstates how much independent
    searching occurred and inflates :math:`SR^*`, making the DSR overly
    conservative.  Supply ``n_effective_trials`` — an estimate of the number
    of *independent* trials (e.g. from clustering the trial return series,
    or :math:`N_{\mathrm{eff}} = 1 + (N-1)(1-\bar{\rho})` for an average
    inter-trial correlation :math:`\bar{\rho}`) — to correct for this.  When
    ``None`` the raw count is used, an upper bound that yields a
    conservative DSR.

    Parameters
    ----------
    sr : float
        Per-period (non-annualised) Sharpe of the *selected* strategy —
        typically ``max(trial_sharpes)``.
    trial_sharpes : sequence of float
        Per-period Sharpe estimates of every trial; their cross-sectional
        variance drives the deflation.
    n : int
        Number of return observations behind ``sr``.
    skew, kurt : float
        Higher moments of the selected strategy's returns (``kurt``
        non-excess), forwarded to :func:`probabilistic_sharpe_ratio`.
    n_effective_trials : int | None
        Effective number of independent trials; defaults to
        ``len(trial_sharpes)``.

    Returns
    -------
    float
        DSR ∈ (0, 1): probability the selected strategy's true Sharpe is
        positive *after* accounting for selection across ``N`` trials.
    """
    trials = np.asarray(trial_sharpes, dtype=float)
    if trials.size < 2:
        raise ValueError("Need at least 2 trial Sharpes to estimate dispersion.")
    n_trials = trials.size if n_effective_trials is None else n_effective_trials
    if n_trials < 2:
        raise ValueError(f"n_effective_trials must be >= 2, got {n_trials}.")

    sr_variance = float(np.var(trials, ddof=1))
    expected_max_z = (1.0 - _EULER_MASCHERONI) * _norm_ppf(
        1.0 - 1.0 / n_trials
    ) + _EULER_MASCHERONI * _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    sr_star = math.sqrt(sr_variance) * expected_max_z
    return probabilistic_sharpe_ratio(
        sr=sr, sr_benchmark=sr_star, n=n, skew=skew, kurt=kurt
    )


def _per_period_sharpe_and_moments(returns: pd.Series) -> tuple[float, int, float, float]:
    """Return ``(sr, n, skew, kurt)`` for PSR/DSR from a per-period return series.

    The Sharpe is **non-annualised** (mean / std), and ``kurt`` is converted
    from pandas' *excess* kurtosis to the **non-excess** convention required
    by :func:`probabilistic_sharpe_ratio` (Gaussian → 3).
    """
    r = returns.dropna()
    n = len(r)
    if n < 2:
        raise ValueError(f"Need at least 2 returns, got {n}.")
    std = float(r.std(ddof=1))
    if std == 0.0:
        raise ValueError("Cannot compute Sharpe: zero return standard deviation.")
    sr = float(r.mean()) / std
    return sr, n, float(r.skew()), float(r.kurt()) + 3.0


def probabilistic_sharpe_ratio_from_returns(
    returns: pd.Series,
    sr_benchmark: float = 0.0,
) -> float:
    """PSR computed directly from a per-period return series.

    Convenience wrapper that derives the per-period Sharpe and the (non-excess)
    higher moments with the correct conventions, then calls
    :func:`probabilistic_sharpe_ratio`.

    Parameters
    ----------
    returns : pd.Series
        Per-period strategy returns.
    sr_benchmark : float
        Per-period benchmark Sharpe to beat (``0.0`` → test SR > 0).
    """
    sr, n, skew, kurt = _per_period_sharpe_and_moments(returns)
    return probabilistic_sharpe_ratio(sr, sr_benchmark, n, skew, kurt)


def deflated_sharpe_ratio_from_returns(
    trial_returns: pd.DataFrame,
    selected: str | None = None,
    n_effective_trials: int | None = None,
) -> float:
    """DSR computed directly from a matrix of trial return series.

    Each column of ``trial_returns`` is one trial's per-period returns.  The
    per-period Sharpe of every column forms the dispersion term; the
    ``selected`` column (default: highest-Sharpe trial) supplies ``sr`` and
    the higher moments passed to :func:`deflated_sharpe_ratio`.

    Parameters
    ----------
    trial_returns : pd.DataFrame
        Columns = trials, rows = periods.  May contain leading NaNs
        (dropped per column).
    selected : str | None
        Column to treat as the chosen strategy.  ``None`` picks the
        highest per-period Sharpe.
    n_effective_trials : int | None
        Effective number of independent trials (see
        :func:`deflated_sharpe_ratio`).  Defaults to the column count.
    """
    if trial_returns.shape[1] < 2:
        raise ValueError("Need at least 2 trial columns to compute a DSR.")
    trial_sharpes = {
        col: _per_period_sharpe_and_moments(trial_returns[col])[0]
        for col in trial_returns.columns
    }
    if selected is None:
        selected = max(trial_sharpes, key=trial_sharpes.get)
    sr, n, skew, kurt = _per_period_sharpe_and_moments(trial_returns[selected])
    return deflated_sharpe_ratio(
        sr=sr,
        trial_sharpes=list(trial_sharpes.values()),
        n=n,
        skew=skew,
        kurt=kurt,
        n_effective_trials=n_effective_trials,
    )


# ---------------------------------------------------------------------------
# TimeseriesAnalyzer
# ---------------------------------------------------------------------------


class TimeseriesAnalyzer:
    """
    Evaluates the performance of one or more financial strategies.

    Parameters
    ----------
    prices : pd.DataFrame
        Each column represents a strategy's indexed performance (NAV / price level).
        Must have a DatetimeIndex. Returns are derived internally via pct_change().
    risk_free_col : str | None
        Name of a column in ``prices`` to use as the dynamic risk-free rate
        benchmark for Sharpe / Sortino calculations.
        If None, a zero series is used as the risk-free rate.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        risk_free_col: str | None = None,
    ) -> None:
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a pd.DataFrame.")
        _validate_datetime_index(prices)

        self.prices: pd.DataFrame = prices.copy()
        self.risk_free_col: str | None = risk_free_col
        self._returns: pd.DataFrame = prices.pct_change()

        if risk_free_col is not None:
            if risk_free_col not in prices.columns:
                raise ValueError(
                    f"risk_free_col '{risk_free_col}' not found in DataFrame columns."
                )
            self._risk_free: pd.Series = self._returns[risk_free_col]
            self._strategy_cols: list[str] = [
                c for c in self._returns.columns if c != risk_free_col
            ]
        else:
            self._risk_free = pd.Series(0.0, index=self._returns.index)
            self._strategy_cols = list(self._returns.columns)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def returns(self) -> pd.DataFrame:
        """Period returns derived from prices via pct_change()."""
        return self._returns

    @property
    def strategy_returns(self) -> pd.DataFrame:
        """Period returns for strategy columns only (excludes risk-free column)."""
        return self._returns[self._strategy_cols]

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def apply_functions(
        self,
        functions: list[Callable | str],
        annualize: bool = True,
    ) -> pd.DataFrame:
        """
        Apply a list of metric functions to all strategy columns.

        Parameters
        ----------
        functions : list
            Each element can be a callable (e.g. ``calc_cagr``) or a string
            name of a function defined in this module (e.g. ``"calc_cagr"``).
        annualize : bool
            Passed through to each metric function.

        Returns
        -------
        pd.DataFrame
            Rows = metric names, Columns = strategy names.
        """
        resolved: list[Callable] = []
        for fn in functions:
            if callable(fn):
                resolved.append(fn)
            elif isinstance(fn, str):
                fn_obj = globals().get(fn)
                if fn_obj is None or not callable(fn_obj):
                    raise ValueError(
                        f"Function '{fn}' not found in timeseries_analyzer module."
                    )
                resolved.append(fn_obj)
            else:
                raise TypeError(f"Expected callable or str, got {type(fn)}.")

        rows: dict[str, dict[str, Any]] = {}
        for fn in resolved:
            result = self._apply_single_fn(fn, annualize=annualize)
            if isinstance(result, pd.Series):
                rows[fn.__name__] = result.to_dict()
            elif isinstance(result, pd.DataFrame):
                # calc_max_drawdown returns a DataFrame; flatten into prefixed rows
                for metric_key in result.index:
                    rows[f"{fn.__name__}__{metric_key}"] = result.loc[
                        metric_key
                    ].to_dict()
            else:
                rows[fn.__name__] = {col: result for col in self._strategy_cols}

        return pd.DataFrame(rows).T[self._strategy_cols]

    def _apply_single_fn(self, fn: Callable, annualize: bool) -> Any:
        """
        Apply a single metric function column-by-column.

        Each column is dropna()'d independently so that staggered start dates
        do not affect other series. The risk-free / benchmark series is
        reindexed to match each column's available dates.
        """
        params = list(inspect.signature(fn).parameters.keys())
        col_results: dict[str, Any] = {}

        for col in self._strategy_cols:
            col_ret: pd.Series = self._returns[col].dropna()
            kwargs: dict[str, Any] = {"annualize": annualize}

            if "risk_free_rate" in params:
                kwargs["risk_free_rate"] = self._risk_free.reindex(col_ret.index)
            elif "benchmark" in params:
                kwargs["benchmark"] = self._risk_free.reindex(col_ret.index)

            col_results[col] = fn(col_ret, **kwargs)

        if fn is calc_max_drawdown:
            return pd.DataFrame(col_results)

        return pd.Series(col_results)

    def apply_standard_functions(self, annualize: bool = True) -> pd.DataFrame:
        """
        Apply the default suite of performance metrics.

        Returns
        -------
        pd.DataFrame
            Rows = metric names, Columns = strategy names.
        """
        return self.apply_functions(
            [
                calc_cagr,
                calc_arithmetic_mean_return,
                calc_volatility,
                calc_sharpe_ratio,
                calc_sharpe_ratio_rf0,
                calc_sortino_ratio,
                calc_max_drawdown,
                calc_var_95,
                calc_var_99,
                calc_cvar_95,
                calc_cvar_99,
            ],
            annualize=annualize,
        )

    def rolling_describe(self, window_years: float = 1.0) -> dict[str, pd.DataFrame]:
        """
        1-year rolling descriptive statistics for each strategy column at native frequency.

        Parameters
        ----------
        window_years : float
            Window size in years; converted to integer periods using inferred frequency.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys = strategy column names.
            Values = DataFrames with columns: count, mean, std, min, 25%, 50%, 75%, max.
        """
        return {
            col: calc_rolling_describe(
                self._returns[col].dropna(), window_years=window_years
            )
            for col in self._strategy_cols
        }

    # ------------------------------------------------------------------
    # Excel output
    # ------------------------------------------------------------------

    def to_excel(
        self,
        path: str,
        sheet_name: str = "Performance",
        annualize: bool = True,
    ) -> None:
        """
        Write performance metrics and indexed performance (NAV) to an Excel workbook.

        Layout
        ------
        Block 1 (starts at A1): performance summary table
            - Row 1  : header  ("Metric" | strategy names)
            - Rows 2+: one row per metric
        [blank row]
        Block 2: indexed performance time-series
            - Row 1  : header  ("Date" | strategy names)
            - Rows 2+: one row per observation

        Parameters
        ----------
        path : str
            Destination .xlsx file path (created or overwritten).
        sheet_name : str
            Target worksheet name.
        annualize : bool
            Passed through to apply_standard_functions().
        """
        try:
            import xlwings as xw  # type: ignore
        except ImportError as e:
            raise ImportError(
                "xlwings is required for spreadsheet output. "
                "Install with: uv add xlwings"
            ) from e

        metrics = self.apply_standard_functions(annualize=annualize)
        performance = self.prices[self._strategy_cols]

        # Build 2-D arrays for bulk write (much faster than row-by-row)
        metrics_block: list[list[Any]] = [
            ["Metric"] + list(metrics.columns),
            *[
                [str(name)] + [_safe_excel_val(v) for v in row]
                for name, row in metrics.iterrows()
            ],
        ]
        performance_block: list[list[Any]] = [
            ["Date"] + list(performance.columns),
            *[
                [str(idx.date())] + [_safe_excel_val(v) for v in row]
                for idx, row in performance.iterrows()
            ],
        ]

        # Block 2 starts two rows below the last metrics row (one blank gap)
        block2_row = len(metrics_block) + 2  # 1-based

        app = xw.App(visible=False)
        try:
            wb = app.books.add()
            ws = wb.sheets[0]
            ws.name = sheet_name

            ws.range("A1").value = metrics_block
            ws.range("1:1").api.Font.Bold = True

            ws.range(f"A{block2_row}").value = performance_block
            ws.range(f"{block2_row}:{block2_row}").api.Font.Bold = True

            ws.autofit("columns")
            wb.save(os.path.abspath(path))
        finally:
            app.quit()


# ---------------------------------------------------------------------------
# Spreadsheet integration (module-level helper)
# ---------------------------------------------------------------------------


def run_excel_report(
    csv_path: str,
    output_xlsx: str = "strategy_analysis.xlsx",
    sheet_name: str = "Performance",
    risk_free_col: str | None = None,
) -> str:
    """
    Load a CSV of strategy prices, compute performance metrics, and write
    results to an Excel workbook.

    Parameters
    ----------
    csv_path : str
        Path to a CSV file with a date index and strategy price columns.
    output_xlsx : str
        Destination .xlsx file path.
    sheet_name : str
        Target worksheet name.
    risk_free_col : str | None
        Column name to use as the risk-free rate benchmark.

    Returns
    -------
    str
        Absolute path to the saved workbook.
    """
    prices: pd.DataFrame = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    analyzer = TimeseriesAnalyzer(prices, risk_free_col=risk_free_col)
    analyzer.to_excel(output_xlsx, sheet_name=sheet_name)
    return os.path.abspath(output_xlsx)


def _safe_excel_val(v: Any) -> Any:
    """Convert non-Excel-safe values (Timestamp, NaT, NaN) to Excel-compatible types."""
    if isinstance(v, pd.Timestamp):
        return str(v.date())
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return v


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Override by passing path as a command-line argument:
    #   uv run analytics/timeseries_analyzer.py "C:\path\to\file.xlsx"
    file_path = r"C:\Users\jakub\gaa_data\performance"
    file_name = "VencelJakub_202602.xlsx"
    file = os.path.join(file_path, file_name)

    print(f"Loading: {file}")
    df = pd.read_excel(file, index_col=0, header=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, format="%d.%m.%Y", errors="coerce")
    df = df.sort_index()
    print(f"Shape: {df.shape}  |  {df.index[0].date()} -> {df.index[-1].date()}\n")

    analyzer = TimeseriesAnalyzer(df)
    metrics = analyzer.apply_standard_functions()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.6f}".format)
    print(metrics.to_string())
    analyzer.to_excel(r"C:\Users\jakub\gaa_data\performance_analysis.xlsx")

    # --- Plot: performance + drawdowns ------------------------------------
    import matplotlib.pyplot as plt

    prices = analyzer.prices
    strategy_cols = [c for c in prices.columns if c != analyzer.risk_free_col]

    fig, (ax_nav, ax_dd) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # Panel 1: NAV (log scale)
    for col in strategy_cols:
        ax_nav.plot(prices.index, prices[col], label=col, linewidth=1.2)
    ax_nav.set_ylabel("NAV (log scale)")
    ax_nav.set_yscale("log")
    ax_nav.set_title("Strategy Performance")
    ax_nav.legend(loc="upper left", fontsize=8)
    ax_nav.grid(True, alpha=0.3)

    # Panel 2: Drawdowns
    for col in strategy_cols:
        cum = prices[col] / prices[col].cummax()
        drawdown = cum - 1.0
        ax_dd.fill_between(drawdown.index, drawdown, 0, alpha=0.3)
        ax_dd.plot(drawdown.index, drawdown, label=col, linewidth=0.8)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_title("Drawdowns")
    ax_dd.legend(loc="lower left", fontsize=8)
    ax_dd.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
