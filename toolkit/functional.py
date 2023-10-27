"""_summary_
"""
import numpy as np
import pandas as pd

PERIODICITY = {
    'D': 252,
    'W': 52,
    'M': 12,
    'Q': 4,
    'A': 1
}

def price_to_return(p: pd.Series)-> pd.Series:
    s = p.pct_change()
    s.index = p.index.to_period()
    return s.dropna()

def return_to_price(r: pd.Series)-> pd.Series:
    s = pd.concat([pd.Series([0]), r])
    s.index = pd.date_range(end=r.index[-1].to_timestamp(how='e').date(), periods=len(r.index) + 1, freq=r.index.freq)
    return (s + 1).cumprod()

# Decorator
def requireReturn(func):
    def wrapper(pre, *args, **kwargs):
        print('Wrapper', args, kwargs)
        post = pre
        if isinstance(pre.index, pd.DatetimeIndex):
            post = price_to_return(pre)
        return func(post, *args, **kwargs)
    return wrapper

@requireReturn
def compound_return(s: pd.Series, annualize=True) -> float:
    return np.exp(np.log1p(s).sum()) - 1

def ytd():
    pass

def mtd():
    pass

def arithmetic_mean(s: pd.Series) -> float:
    return s.mean()

def geometric_mean(s: pd.Series) -> float:
    return (compound_return(s) + 1) ** (1 / len(s))

def avg_pos():
    pass

def avg_neg():
    pass

def volatility(s: pd.Series, annualize=True):
    # Degree of freedom is N-1 for Pandas but N for NumPy
    return s.std()

def vol_pos():
    pass

def vol_neg():
    pass

def skew(s: pd.Series, annualize=True):
    # Degree of freedom is N-1 for Pandas but N for NumPy
    return s.skew()

def kurt(s: pd.Series, annualize=True):
    # Excess kurtosis, SciPy does not correct for bias by default
    return s.kurt()

def var_historical():
    pass

def var_normal():
    pass

def var_modified():
    pass

def expected_shortfall():
    pass

def worst_drawdown():
    pass

def drawdowns():
    pass

def max_upturn():
    pass

def sharpe():
    pass

def sortino():
    pass

def downside_deviation():
    pass

def sterling():
    pass

def calmar():
    pass

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