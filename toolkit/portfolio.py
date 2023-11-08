"""Portfolio Optimization Module

`weights` should be a Series with shape (m,) or a DataFrame with shape (m, k)
`returns` should be a Series with shape (m,) or a DataFrame with shape (n, m)

where:
    n is the number of periods
    m is the number of asset classes
    k is the number of weighting schemes
"""
import numpy as np
import pandas as pd
from scipy import optimize


def portfolio_return(weights: pd.Series | pd.DataFrame, returns: pd.Series | pd.DataFrame) -> np.float64 | pd.Series | pd.DataFrame:
    """Return(s) of the portfolio(s) based on input weight(s).

    Parameters
    ----------
    weights : pd.Series | pd.DataFrame
        Portfolio weights
    returns : pd.Series | pd.DataFrame
        Return(s) of each asset class over single or multiple period(s)

    Returns
    -------
    np.float64 | pd.Series | pd.DataFrame
        Portfolio returns. See below for details.


    When weights is a Series (i.e. Single weighting scheme)
    -------------------------------------------------------
    If a Series of asset returns of a single period is passed, it outputs
    the return of buy-and-hold portfolio as a float (no rebancing).

    If a DataFrame of asset returns over a period is passed, it outputs
    the time series of the of a constant-mix portfolio (rebalance every
    period) as a Series with shape `(n,)`

    When weights is a DataFrame (i.e. Multiple weighting schemes):
    --------------------------------------------------------------
    If a Series of asset returns of a single period is passed, it outputs
    the return of k buy-and-hold portfolios as a Series with shape `(k,)`.

    If a DataFrame of asset returns over a period is passed, it outputs
    the time series of the of k constant-mix portfolios (rebalance every
    period) as a DataFrame with shape `(n, k)`

    Note
    ----
    Index of the weights must match that of the returns.
    """
    return (weights.T @ returns.T).T


def portfolio_volatility(weights: pd.Series | pd.DataFrame, cov: pd.DataFrame) -> np.float64 | pd.Series:
    """Volatility of the portfolio(s) based on input weight(s).

    Parameters
    ----------
    weights : pd.Series | pd.DataFrame
        Portfolio weights
    cov : pd.DataFrame
        Coveriance matrix

    Returns
    -------
    np.float64 | pd.Series
        Portfolio volatilities. See below for details.


    When weights is a Series (i.e. Single weighting scheme)
    -------------------------------------------------------
    It outputs the portfolio volatility as a float.

    When weights is a DataFrame (i.e. Multiple weighting schemes):
    --------------------------------------------------------------
    It outputs the volatilities of each portfolio as a Series with shape `(k,)`

    Note: This method assumes the portfolio is buy-and-hold. To calculate the
    volatility of a constant-mix portfolio, please use `portfolio_return` to
    generate the portfolio returns and then call `volatility`.
    """
    matrix = weights.T @ cov @ weights
    if isinstance(weights, pd.DataFrame):
        matrix = pd.Series(np.diag(matrix), index=matrix.index)
    return np.sqrt(matrix)


def risk_contribution(weights: pd.Series | pd.DataFrame, cov: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Risk contribution of each asset in the portfolio.
    Sum to 100%.

    Parameters
    ----------
    weights : pd.Series | pd.DataFrame
        Portfolio weights
    cov : pd.DataFrame
        Coveriance matrix

    Returns
    -------
    pd.Series | pd.DataFrame
        Risk contribution of each asset
    """
    return (weights.T @ cov * weights.T).div(portfolio_volatility(weights, cov) ** 2, axis=0)

#####################
# Weighting schemes #
#####################


def equal_weight(er: pd.Series) -> pd.Series:
    """Equal weight each asset in a portfolio.
    Also known as naive diversification.

    Parameters
    ----------
    er : pd.Series
        Expected returns

    Returns
    -------
    pd.Series
        Weights of each asset
    """
    return pd.Series(np.ones_like(er) / len(er), index=er.index)


def inverse_vol(cov: pd.DataFrame) -> pd.Series:
    """Weight each asset based on the inverse of its volatility.
    It is sometimes called poor man's risk parity.

    Parameters
    ----------
    cov : pd.DataFrame
        Coveriance matrix

    Returns
    -------
    pd.Series
        Weight of each asset
    """
    inverse = 1. / pd.Series(np.diag(cov), index=cov.index) ** 0.5
    return (inverse / inverse.sum())


def max_return(er: pd.Series) -> pd.Series:
    """Allocate 100% to the asset that generates the highest return, and 0%
    to the rest.

    Parameters
    ----------
    er : pd.Series
        Expected returns

    Returns
    -------
    pd.Series
        Weight of each asset
    """
    wtg = pd.Series(0, index=er.index)
    wtg.loc[er.idxmax()] = 1.
    return wtg


def max_sharpe(er: pd.Series, cov: pd.DataFrame, rfr: float = 0., min: float = float('-inf'), max: float = float('inf')) -> pd.Series:
    """Weight of each asset in the portfolio that achieve the highest Sharpe
    ratio yet satisfy all constraints like no shorting and no leverage.

    Periodicity of risk-free rate should match that of the expected return.
    If expected returns are annualized, risk-free rate should also be
    annualized.

    Parameters
    ----------
    er : pd.Series
        Expected returns
    cov : pd.DataFrame
        Coveriance matrix
    rfr : float, optional
        Risk-free rate, by default 0.
    min : float, optional
        Minimum weight, by default float('-inf') i.e. allow shorting. Set to 0 if no shorting is allowed.
    max : float, optional
        Maximum weight, by default float('inf') i.e. allow leverage. Set to 1 if no leverage is allowed.

    Returns
    -------
    pd.Series
        Weight of each asset
    """
    # Objective function to minimize negative Sharpe (i.e. to maximize Sharpe)
    def neg_sharpe(w, er, cov, rfr):
        return (rfr - portfolio_return(w, er)) / portfolio_volatility(w, cov)
    fully_invest = {'type': 'eq',
                    'fun': lambda w: np.sum(w) - 1}
    wtg = optimize.minimize(neg_sharpe,
                            equal_weight(er),
                            args=(er, cov, rfr),
                            method='SLSQP',
                            options={'disp': False},
                            constraints=(fully_invest,),
                            bounds=((min, max),) * len(er)
                            )
    return wtg.x if wtg.success else np.full_like(er, np.nan)


def min_vol(cov: pd.DataFrame) -> pd.Series:
    """Weight of each asset in the global minimum volatility portfolio.

    Parameters
    ----------
    cov : pd.DataFrame
        Coveriance matrix

    Returns
    -------
    pd.Series
        Weight of each asset
    """
    return max_sharpe(pd.Series(np.repeat(1., cov.shape[0]), index=cov.index), cov, rfr=0)


def risk_parity(cov: pd.DataFrame) -> pd.Series:
    """Weight of each asset in an equal risk contribution portfolio.

    Parameters
    ----------
    cov : pd.DataFrame
        Coveriance matrix

    Returns
    -------
    pd.Series
        Weight of each asset
    """
    # Squared difference
    def msd_risk(w, cov, target):
        return ((risk_contribution(w, cov) - target) ** 2).sum()
    fully_invest = {'type': 'eq',
                    'fun': lambda w: np.sum(w) - 1}
    ew = equal_weight(cov.iloc[0])
    wtg = optimize.minimize(msd_risk,
                            ew,
                            args=(cov, ew),
                            method='SLSQP',
                            options={'disp': False},
                            constraints=(fully_invest,),
                            bounds=((0, 1),) * cov.shape[0]
                            )
    return wtg.x if wtg.success else np.repeat(np.nan, cov.shape[0])


def min_vol_at(target: float, er: pd.Series, cov: pd.DataFrame, min: float = float('-inf'), max: float = float('inf')) -> pd.Series:
    """Weight of each asset in an optimal portfolio that generats the target
    return but has the smallest volatility.

    This method can be used to generate the efficient frontier.

    Parameters
    ----------
    target : float
        Target return
    er : pd.Series
        Expected returns
    cov : pd.DataFrame
        Coveriance matrix
    min : float, optional
        Minimum weight, by default float('-inf') i.e. allow shorting. Set to 0 if no shorting is allowed.
    max : float, optional
        Maximum weight, by default float('inf') i.e. allow leverage. Set to 1 if no leverage is allowed.

    Returns
    -------
    pd.Series
        Weight of each asset
    """
    fully_invest = {'type': 'eq',
                    'fun': lambda w: np.sum(w) - 1}
    target_return = {'type': 'eq',
                     'fun': lambda w: portfolio_return(w, er) - target}
    wtg = optimize.minimize(portfolio_volatility,
                            equal_weight(er),
                            args=(cov),
                            method='SLSQP',
                            options={'disp': False},
                            constraints=(target_return, fully_invest,),
                            bounds=((min, max),) * len(er)
                            )
    return wtg.x if wtg.success else np.full_like(er, np.nan)
