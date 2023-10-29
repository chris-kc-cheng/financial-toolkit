"""Works for both Series and DataFrame
"""
import numpy as np
import scipy
import pandas as pd

PERIODICITY = {
    'D': 252,
    'W': 52,
    'M': 12,
    'Q': 4,
    'A': 1
}

def periodicity(r):
    return PERIODICITY[r.index.freqstr[0]]

def price_to_return(p: pd.Series | pd.DataFrame)-> pd.Series | pd.DataFrame:
    s = p.pct_change()
    s.index = p.index.to_period()
    return s.dropna()

def return_to_price(r: pd.Series | pd.DataFrame)-> pd.Series | pd.DataFrame:    
    if isinstance(r, pd.Series):
        s = pd.concat([pd.Series([0]), r])
    else:
        s = pd.concat([pd.DataFrame(np.zeros((1, r.shape[1]))), r])
    s.index = pd.date_range(end=r.index[-1].to_timestamp(how='e').date(), periods=len(r.index) + 1, freq=r.index.freq)
    return (s + 1).cumprod()

# Decorators
def requireReturn(func):
    def wrapper(pre, *args, **kwargs):
        post = pre
        if isinstance(pre.index, pd.DatetimeIndex):
            post = price_to_return(pre)
        return func(post, *args, **kwargs)
    return wrapper

def requirePrice(func):
    def wrapper(pre, *args, **kwargs):
        post = pre
        if isinstance(pre.index, pd.PeriodIndex):
            post = return_to_price(pre)
        return func(post, *args, **kwargs)
    return wrapper

@requireReturn
def compound_return(s: pd.Series | pd.DataFrame, annualize=False) -> float | pd.Series:
    r = np.exp(np.log1p(s).sum())
    if annualize:
        r **= periodicity(s) / len(s)
    return r - 1

@requireReturn
def arithmetic_mean(s: pd.Series | pd.DataFrame) -> float | pd.Series:
    return s.mean()

# FIXME not tested yet @requireReturn

def geometric_mean(s: pd.Series | pd.DataFrame) -> float | pd.Series:
    return (compound_return(s, False) + 1) ** (1 / len(s)) - 1

@requireReturn
def mean_abs_dev(s: pd.Series | pd.DataFrame) -> float | pd.Series:
    mean = s.mean()
    return np.absolute(s - mean).sum() / len(s)

@requireReturn
def variance(s: pd.Series | pd.DataFrame, annualize=False) -> float | pd.Series:
    v = s.var()
    if annualize:
        v *= periodicity(s)
    return v

def avg_pos(s: pd.Series) -> float:
    return s[s >= 0].mean()

def avg_neg(s: pd.Series) -> float:
    return s[s < 0].mean()

def vol_pos(s: pd.Series) -> float:
    return s[s >= 0].std()

def vol_neg(s: pd.Series) -> float:
    return s[s < 0].std()

@requireReturn
def volatility(s: pd.Series | pd.DataFrame, annualize=False):
    # Degree of freedom is N-1 for Pandas but N for NumPy
    v = s.std()
    if annualize:
        v *= np.sqrt(periodicity(s))
    return v

@requireReturn
def skew(s: pd.Series | pd.DataFrame):
    # Degree of freedom is N-1 for Pandas but N for NumPy
    return s.skew()

@requireReturn
def kurt(s: pd.Series | pd.DataFrame):
    # Excess kurtosis, SciPy does not correct for bias by default
    return s.kurt()

@requireReturn
def covariance(df = pd.DataFrame) -> pd.DataFrame:
    return df.cov()

@requireReturn
def correlation(df = pd.DataFrame) -> pd.DataFrame:
    return df.corr()

@requireReturn
def sharpe(s: pd.Series | pd.DataFrame, rfr: float = 0, annualize=True) -> float | pd.Series:
    # rfr is annual
    rate = rfr
    if not annualize:
        rate = (1 + rfr) ** (1 / periodicity(s))
    return (compound_return(s, annualize) - rate) / volatility(s, annualize)

#FIXME not tested @requirePrice
def max_upturn(p: pd.Series | pd.DataFrame) -> float | pd.Series:
    return (p / p.cummin()).max()

@requirePrice
def worst_drawdown(p: pd.Series | pd.DataFrame) -> float | pd.Series:
    return (p / p.cummax()).min() - 1

@requirePrice
def all_drawdown(p: pd.Series | pd.DataFrame) -> float | pd.Series:
    m = p.cummax()
    peak = (p == m) & (p < m).shift(-1)
    num = peak.cumsum()
    dd = p.groupby(num).aggregate(worst_drawdown)
    return dd[dd < 0].sort_values()

@requireReturn
def avg_annual_drawdown(p: pd.Series | pd.DataFrame) -> float | pd.Series:
    return p.groupby(p.index.year).aggregate(worst_drawdown).mean()

def avg_drawdown(p: pd.Series | pd.DataFrame, d: int = 3) -> float | pd.Series:
    return all_drawdown(p)[:d].mean()

def calmar(s: pd.Series | pd.DataFrame, rfr: float = 0) -> float | pd.Series:
    return (compound_return(s, True) - rfr) / -worst_drawdown(s)

def sterling(s: pd.Series | pd.DataFrame) -> float | pd.Series:
    return compound_return(s, True) / np.absolute(avg_annual_drawdown() - 0.1)

def sterling_modified(s: pd.Series | pd.DataFrame, rfr: float = 0, d: int = 3) -> float | pd.Series:
    return (compound_return(s, True) - rfr) / np.absolute(avg_drawdown(s, d))

@requireReturn
def drawdown_deviation(s: pd.Series | pd.DataFrame, d: int = 3) -> float | pd.Series:
    return -np.sqrt((all_drawdown(s)[:d] ** 2).sum() / (len(s)))

def burke_modified(s: pd.Series | pd.DataFrame, rfr: float = 0, d: int = 3) -> float | pd.Series:
    return (compound_return(s, True) - rfr) / -drawdown_deviation(s, d)

def sterling_calmar(s: pd.Series | pd.DataFrame, rfr: float = 0, d: int = 3) -> float | pd.Series:
    return (compound_return(s, True) - rfr) / -avg_annual_drawdown(s)

@requirePrice
def underwater(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return s / s.cummax() - 1

@requireReturn
# Positive
def ulcer_index(s: pd.Series | pd.DataFrame) -> float | pd.Series:
    return np.sqrt((underwater(s) ** 2).sum() / len(s))

@requireReturn
# Positive
def pain_index(s: pd.Series | pd.DataFrame) -> float | pd.Series:
    return np.absolute(underwater(s)).sum() / len(s)

def martin(s: pd.Series | pd.DataFrame, rfr: float = 0) -> float | pd.Series:
    return (compound_return(s, True) - rfr) / ulcer_index(s)

def pain(s: pd.Series | pd.DataFrame, rfr: float = 0) -> float | pd.Series:
    return (compound_return(s, True) - rfr) / pain_index(s)

@requireReturn
def downside_potential(s: pd.Series | pd.DataFrame, mar : float = 0) -> float | pd.Series:
    return (mar - s[s < mar]).sum() / len(s)

@requireReturn
def upside_potential(s: pd.Series | pd.DataFrame, mar : float = 0) -> float | pd.Series:
    return (s[s > mar] - mar).sum() / len(s)

@requireReturn
def downside_risk(s: pd.Series | pd.DataFrame, mar : float = 0, annualize=False) -> float | pd.Series:
    dr = np.sqrt(((mar - s[s < mar]) ** 2).sum() / len(s))
    if annualize:
        dr *= np.sqrt(periodicity(s))
    return dr

@requireReturn
def upside_risk(s: pd.Series | pd.DataFrame, mar : float = 0, annualize=False) -> float | pd.Series:
    ur = np.sqrt(((s[s > mar] - mar) ** 2).sum() / len(s))
    if annualize:
        ur *= np.sqrt(periodicity(s))
    return ur

def omega(s: pd.Series | pd.DataFrame, mar : float = 0) -> float:
    return upside_potential(s, mar) / downside_potential(s, mar)

def sortino(s: pd.Series | pd.DataFrame, mar : float = 0) -> float | pd.Series:
    ann_mar = (1 + mar) ** periodicity(s) - 1
    return (compound_return(s, annualize=True) - ann_mar ) / downside_risk(s, mar, annualize=True)

def upside_potential_ratio(s: pd.Series | pd.DataFrame, mar : float = 0) -> float | pd.Series:
    return upside_potential(s, mar) / downside_risk(s, mar, annualize=True)

def variability_skewness(s: pd.Series | pd.DataFrame, mar : float = 0) -> float | pd.Series:
    return upside_risk(s, mar) / downside_risk(s, mar)

# TODO:

def summary(s: pd.Series | pd.DataFrame):
    return {'Skeweness': skew(s),
            'Kurtosis' : kurt(s)}

def var_historical(s: pd.Series, z: float = 0.01):
    return np.percentile(s, z)

def cvar_historical(s: pd.Series, z: float = 0.01):
    return s[s < var_historical(s, z)].mean()

def is_normal(s: pd.Series, z: float = 0.01):
    return scipy.stats.jarque_bera(s)[1] > z

def var_normal(s: pd.Series, z: float = 0.95, annualize=False) -> float:
    z = abs(scipy.stats.norm.ppf(0.95))
    mu = compound_return(s, annualize)
    sigma = volatility(s, annualize)
    return mu - sigma * z
    
# Accept both 5% or 95%
def var_modified(s: pd.Series, z: float = 0.95, annualize=False) -> float:
    z = abs(scipy.stats.norm.ppf(0.95))
    mu = compound_return(s, annualize)
    sigma = volatility(s, annualize)
    S = skew(s)
    K = kurt(s)
    t = z + (z ** 2 - 1) * S / 6 + (z ** 3 - 3 * z) * K / 24 - (2 * z ** 3 - 5 * z) * S ** 2 / 36
    return mu - sigma * t

def cvar_normal(s: pd.Series, z: float = 0.01, annualize=False):
    mu = compound_return(s, annualize)
    sigma = volatility(s, annualize)
    return mu - sigma * scipy.stats.norm.pdf(scipy.stats.norm.ppf(z)) / z

def ytd():
    pass

def mtd():
    pass

def drawdowns():
    pass

# Relative to benchmark
def alpha():
    pass

def beta():
    pass

def betas():
    pass

def corr():
    pass

def tracking_error():
    pass

def information_ration():
    pass

def up_capture():
    pass

def down_capture():
    pass

def treynor():
    pass