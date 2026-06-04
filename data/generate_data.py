"""
generate_data.py
----------------
Minimal data helper for the course notebooks: appraisal-smoothing inversion
(Geltner et al. 1994) used in the Yale case study to unsmooth illiquid-asset
returns.
"""

import pandas as pd
import statsmodels.api as sm


def unsmooth_Geltner(returns: pd.Series, max_beta: float = None) -> pd.Series:
    """
    Unsmooth a returns series by inverting an AR(1) appraisal-smoothing filter
    following the approach in Geltner et al. (1994).


    Returns
    - pd.Series
        Unsmooothed returns aligned to the input index. The first observation will
        be NaN because the AR(1) inversion requires a lag.

    Raises
    - ValueError if there are fewer than 2 non-NaN observations to estimate rho.
    """

    if not isinstance(returns, pd.Series):
        raise TypeError("`returns` must be a pandas Series")

    # Work on a copy and preserve original index
    r = returns.copy()
    if r.dropna().shape[0] < 2:
        raise ValueError("Need at least two non-NA observations to unsmooth")

    # estimate rho if not provided
    y = r.dropna()
    y_lag = y.shift(1).dropna()
    common_idx = y.index.intersection(y_lag.index)
    y_curr = y.loc[common_idx]
    X_curr = y_lag.loc[common_idx]
    # estimate AR(1) coefficient without intercept per Geltner (1994)
    model = sm.OLS(y_curr, X_curr)
    res = model.fit()
    # single-parameter model: beta is the first (and only) param
    beta = float(res.params.iloc[0])

    if abs(1.0 - beta) < 1e-12:
        raise ValueError(
            "Estimated beta is numerically equal to 1. Unsmoothing unstable."
        )

    if max_beta is not None:
        beta = min(beta, max_beta)

    # Unsmoothing as inversion of exponential smoothing (Geltner 1994):
    # u_t = (r_t - beta * r_{t-1}) / (1 - beta)

    r_unsmoothed = (r - beta * r.shift(1)) / (1.0 - beta)

    r_unsmoothed.name = returns.name
    return r_unsmoothed, beta
