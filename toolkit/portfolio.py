"""_summary_

Assumptions:
    n Periods
    m Asset classes
    k weighting schemes

"""
import numpy as np
import pandas as pd
from scipy import optimize

# Pass er assume buy-and-hold, pass return series assume constant-mix (rebalance)
# Note index name/type must match
def portfolio_return(weights: pd.Series | pd.DataFrame, returns: pd.Series | pd.DataFrame) -> np.float64 | pd.Series | pd.DataFrame:
    """_summary_

    4 Scenarios
    ===========
    Weight is Series, Returns is Series --> float64
    Weight is DataFrame, Returns is Series --> Series of shape (k,)
    Weight is Series, Returns is DataFrame --> Series of shape (n,)
    Weight is DataFrame, Returns is DataFrame --> DataFrame of shape (n, k)

    Args:
        weights (pd.Series | pd.DataFrame): _description_
        returns (pd.Series | pd.DataFrame): _description_

    Returns:
        np.float64 | pd.Series | pd.DataFrame: _description_
    """
    return (weights.T @ returns.T).T

def portfolio_volatility(weights: pd.Series | pd.DataFrame, cov: pd.DataFrame) -> np.float64 | pd.Series:
    """_summary_

    2 Scenarios
    ===========
    Weight is Series --> float64
    Weight is DataFrame --> Series of shape (k,)

    Args:
        weights (pd.Series | pd.DataFrame): _description_
        cov (pd.DataFrame): _description_

    Returns:
        np.float64 | pd.Series: _description_
    """
    matrix = weights.T @ cov @ weights
    if isinstance(weights, pd.DataFrame):
        matrix = pd.Series(np.diag(matrix), index=matrix.index)
    return np.sqrt(matrix)

def risk_contribution(weights: pd.Series | pd.DataFrame, cov: pd.DataFrame) -> pd.Series | pd.DataFrame:
    """_summary_

    2 Scenarios
    ===========
    Weight is Series --> Series of shape (m,)
    Weight is DataFrame --> DataFrame of shape (k, m)

    Args:
        weights (pd.Series | pd.DataFrame): _description_
        cov (pd.DataFrame): _description_

    Returns:
        pd.Series | pd.DataFrame: _description_
    """
    return (weights.T @ cov * weights.T).div(portfolio_volatility(weights, cov) ** 2, axis=0)

# Weighting schemes
def equal_weight(er: pd.Series) -> pd.Series:
    return pd.Series(np.ones_like(er) / len(er), index=er.index)

def inverse_vol(cov: pd.DataFrame) -> pd.Series:
    inverse = 1. / pd.Series(np.diag(cov), index=cov.index) ** 0.5
    return (inverse / inverse.sum())

def max_return(er: pd.Series) -> pd.Series:
    wtg = pd.Series(0, index=er.index)
    wtg.loc[er.idxmax()] = 1.
    return wtg

def max_sharpe(er: pd.Series, cov: pd.DataFrame, rfr: float = 0., min: float = float('-inf'), max: float = float('inf')) -> pd.Series:
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

def min_vol(cov: pd.DataFrame) -> np.ndarray:
    return max_sharpe(pd.Series(np.repeat(1., cov.shape[0]), index=cov.index), cov, rfr=0)

def min_vol_at(target: float, er: pd.Series, cov: pd.DataFrame, min: float = float('-inf'), max: float = float('inf')) -> pd.Series:
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

def risk_parity(cov: pd.DataFrame) -> pd.Series:
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
