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
        print('Wrapper', args, kwargs)
        post = pre
        if isinstance(pre.index, pd.DatetimeIndex):
            post = price_to_return(pre)
        return func(post, *args, **kwargs)
    return wrapper

def requirePrice(func):
    def wrapper(pre, *args, **kwargs):
        print('Wrapper', args, kwargs)
        post = pre
        if isinstance(pre.index, pd.PeriodIndex):
            post = return_to_price(pre)
        return func(post, *args, **kwargs)
    return wrapper

@requireReturn
def compound_return(s: pd.Series, annualize=False) -> float:
    r = np.exp(np.log1p(s).sum())
    if annualize:
        r **= periodicity(s) / len(s)
    return r - 1

def arithmetic_mean(s: pd.Series) -> float:
    return s.mean()

def geometric_mean(s: pd.Series) -> float:
    return (compound_return(s, False) + 1) ** (1 / len(s)) - 1

def avg_pos(s: pd.Series) -> float:
    return s[s >= 0].mean()

def avg_neg(s: pd.Series) -> float:
    return s[s < 0].mean()

def vol_pos(s: pd.Series) -> float:
    return s[s >= 0].std()

def vol_neg(s: pd.Series) -> float:
    return s[s < 0].std()

def volatility(s: pd.Series, annualize=True):
    # Degree of freedom is N-1 for Pandas but N for NumPy
    v = s.std()
    if annualize:
        v **= np.sqrt(periodicity(s))
    return v

def skew(s: pd.Series):
    # Degree of freedom is N-1 for Pandas but N for NumPy
    return s.skew()

def kurt(s: pd.Series):
    # Excess kurtosis, SciPy does not correct for bias by default
    return s.kurt()

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

@requirePrice
def max_upturn(p: pd.Series) -> float:
    return (p / p.cummin()).max()

@requirePrice
def worst_drawdown(p: pd.Series) -> float:
    return (p / p.cummax()).min() - 1

def drawdowns():
    pass

def sharpe(s: pd.Series, rfr: float = 0, annualize=False) -> float:
    if not annualize:
        r = (1 + rfr) ** (1 / periodicity(p))
    return (compound_return(s, annualize) - r) / volatility(s, annualize)

def downside_deviation(s: pd.Series, mar : float = 0) -> float:
    return (s[s > mar] ** 2).sum() / len(p)

def sortino(s: pd.Series, mar : float = 0) -> float:
    return (compound_return(s) - mar) / downside_deviation(s)

def sterling():
    pass

# Drawdown ratio
def calmar():
    pass

# Winning vs Losing
def omega():
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