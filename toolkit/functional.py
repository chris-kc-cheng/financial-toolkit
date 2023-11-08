"""Works for both Series and DataFrame
"""
from functools import wraps
import numpy as np
import scipy
import pandas as pd
import statsmodels.api as sm

PERIODICITY = {
    'D': 365,
    'B': 252,
    'W': 52,
    'M': 12,
    'Q': 4,
    'A': 1
}


def periodicity(r):
    return PERIODICITY[r.index.freqstr[0]]


def price_to_return(p: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    freq = p.index.freqstr
    s = p.pct_change(fill_method=None)
    s.index = p.index.to_period('D' if freq == 'B' else freq)
    return s.dropna()


def return_to_price(r: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(r, pd.Series):
        s = pd.concat([pd.Series([0]), r])
    else:
        s = pd.concat(
            [pd.DataFrame(np.zeros((1, r.shape[1])), columns=r.columns), r])
    s.index = pd.date_range(
        end=r.index[-1].to_timestamp(how='e').date(), periods=len(r.index) + 1, freq=r.index.freq)
    return (s + 1).cumprod()

# Decorators


def requireReturn(func):
    """Convert to return if input is price
    i.e. Index is DatetimeIndex

    Args:
        func (_type_): _description_
    """
    @wraps(func)
    def wrapper(pre, *args, **kwargs):
        post = pre
        if isinstance(pre.index, pd.DatetimeIndex):
            post = price_to_return(pre)
        return func(post, *args, **kwargs)
    wrapper.__doc__ = func.__doc__
    return wrapper


def requirePrice(func):
    """Convert to price if input is return
    i.e. Index is a PeriodIndex

    Args:
        func (_type_): _description_
    """
    @wraps(func)
    def wrapper(pre, *args, **kwargs):
        post = pre
        if isinstance(pre.index, pd.PeriodIndex):
            post = return_to_price(pre)
        return func(post, *args, **kwargs)
    wrapper.__doc__ = func.__doc__
    return wrapper


def requireBenchmark(func):
    @wraps(func)
    def wrapper(x, pre, *args, **kwargs):
        post = pre
        if isinstance(pre.index, pd.DatetimeIndex):
            post = price_to_return(pre)
        return func(x, post, *args, **kwargs)
    return wrapper


def convertFX(s: pd.Series | pd.DataFrame, fc: pd.Series, lc: pd.Series):
    # Yahoo always use direct quote (vs Bloomberg use indirect quote for GBP/EUR/AUD/NZD)
    if type(s.index) == type(fc.index) and type(s.index) == type(lc.index) and periodicity(s) <= periodicity(fc) and periodicity(s) <= periodicity(lc):
        if isinstance(s.index, pd.DatetimeIndex):
            # Price
            _fc = fc.reindex(s.index).ffill()
            _lc = lc.reindex(s.index).ffill()
            return s.div(_fc, axis=0).mul(_lc, axis=0)
        else:
            # Return
            return convertFX(return_to_price(s), return_to_price(fc), return_to_price(lc))
    else:
        # Error converting, returning original
        return s


@requireReturn
def compound_return(ts: pd.Series | pd.DataFrame, annualize=False) -> float | pd.Series:
    """
    Compound return of time series


    Args:
        ts (pd.Series | pd.DataFrame): Pandas Series or DataFrame of returns
        annualize (bool, optional): Annualizing the compound return. Defaults
        to False.

    Returns:
        float | pd.Series: _description_

    Examples:
        >>> 1 + 2 = 3
    """
    r = np.exp(np.log1p(ts).sum(min_count=1))
    if annualize:
        r **= periodicity(ts) / len(ts)
    return r - 1


@requireReturn
def arithmetic_mean(s: pd.Series | pd.DataFrame) -> float | pd.Series:
    return s.mean()


@requireReturn
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
def covariance(df=pd.DataFrame, annualize=False) -> pd.DataFrame:
    cov = df.cov()
    if annualize:
        cov *= periodicity(df)
    return cov


@requireReturn
def correlation(df=pd.DataFrame) -> pd.DataFrame:
    return df.corr()


@requireReturn
def sharpe(s: pd.Series | pd.DataFrame, rfr_annualized: float = 0, annualize=True) -> float | pd.Series:
    rate = rfr_annualized
    if not annualize:
        rate = (1 + rfr_annualized) ** (1 / periodicity(s))
    return (compound_return(s, annualize) - rate) / volatility(s, annualize)

# FIXME not tested @requirePrice


def max_upturn(p: pd.Series | pd.DataFrame) -> float | pd.Series:
    return (p / p.cummin()).max()


@requirePrice
def worst_drawdown(ts: pd.Series | pd.DataFrame) -> float | pd.Series:
    """Get the worst drawdown of time series

    Args:
        ts (pd.Series | pd.DataFrame): _description_

    Returns:
        float | pd.Series: The max drawdown which is always a negative number
    """
    return (ts / ts.cummax()).min() - 1


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


def calmar(s: pd.Series | pd.DataFrame, rfr_annualized: float = 0) -> float | pd.Series:
    return (compound_return(s, True) - rfr_annualized) / -worst_drawdown(s)


def sterling(s: pd.Series | pd.DataFrame) -> float | pd.Series:
    return compound_return(s, True) / np.absolute(avg_annual_drawdown() - 0.1)


def sterling_modified(s: pd.Series | pd.DataFrame, rfr_annualized: float = 0, d: int = 3) -> float | pd.Series:
    return (compound_return(s, True) - rfr_annualized) / np.absolute(avg_drawdown(s, d))


@requireReturn
def drawdown_deviation(s: pd.Series | pd.DataFrame, d: int = 3) -> float | pd.Series:
    return -np.sqrt((all_drawdown(s)[:d] ** 2).sum() / (len(s)))


def burke_modified(s: pd.Series | pd.DataFrame, rfr_annualized: float = 0, d: int = 3) -> float | pd.Series:
    return (compound_return(s, True) - rfr_annualized) / -drawdown_deviation(s, d)


def sterling_calmar(s: pd.Series | pd.DataFrame, rfr_annualized: float = 0, d: int = 3) -> float | pd.Series:
    return (compound_return(s, True) - rfr_annualized) / -avg_annual_drawdown(s)


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


def martin(s: pd.Series | pd.DataFrame, rfr_annualized: float = 0) -> float | pd.Series:
    return (compound_return(s, True) - rfr_annualized) / ulcer_index(s)


def pain(s: pd.Series | pd.DataFrame, rfr_annualized: float = 0) -> float | pd.Series:
    return (compound_return(s, True) - rfr_annualized) / pain_index(s)


@requireReturn
def downside_potential(s: pd.Series | pd.DataFrame, mar: float = 0) -> float | pd.Series:
    return (mar - s[s < mar]).sum() / len(s)


@requireReturn
def upside_potential(s: pd.Series | pd.DataFrame, mar: float = 0) -> float | pd.Series:
    return (s[s > mar] - mar).sum() / len(s)


@requireReturn
def downside_risk(s: pd.Series | pd.DataFrame, mar: float = 0, annualize=False) -> float | pd.Series:
    dr = np.sqrt(((mar - s[s < mar]) ** 2).sum() / len(s))
    if annualize:
        dr *= np.sqrt(periodicity(s))
    return dr


@requireReturn
def upside_risk(s: pd.Series | pd.DataFrame, mar: float = 0, annualize=False) -> float | pd.Series:
    ur = np.sqrt(((s[s > mar] - mar) ** 2).sum() / len(s))
    if annualize:
        ur *= np.sqrt(periodicity(s))
    return ur


def omega(s: pd.Series | pd.DataFrame, mar: float = 0) -> float:
    return upside_potential(s, mar) / downside_potential(s, mar)


def sortino(s: pd.Series | pd.DataFrame, mar: float = 0) -> float | pd.Series:
    ann_mar = (1 + mar) ** periodicity(s) - 1
    return (compound_return(s, annualize=True) - ann_mar) / downside_risk(s, mar, annualize=True)


def upside_potential_ratio(s: pd.Series | pd.DataFrame, mar: float = 0) -> float | pd.Series:
    return upside_potential(s, mar) / downside_risk(s, mar, annualize=True)


def variability_skewness(s: pd.Series | pd.DataFrame, mar: float = 0) -> float | pd.Series:
    return upside_risk(s, mar) / downside_risk(s, mar)

# Additional

@requireReturn
def best_period(s: pd.Series) -> float:
    return s.max()

@requireReturn
def worst_period(s: pd.Series) -> float:
    return s.min()

@requireReturn
def avg_pos(s: pd.Series) -> float:
    return s[s >= 0].mean()

@requireReturn
def avg_neg(s: pd.Series) -> float:
    return s[s < 0].mean()

@requireReturn
def vol_pos(s: pd.Series) -> float:
    return s[s >= 0].std()

@requireReturn
def vol_neg(s: pd.Series) -> float:
    return s[s < 0].std()

#########################
# Relative to benchmark #
#########################
# Benchmark is already net of risk-free rate
# If risk-free rate is yet to substract from s, you must supply the rfr_periodic


@requireReturn
@requireBenchmark
def beta(s: pd.Series | pd.DataFrame, benchmark: pd.Series | pd.DataFrame, rfr_periodic: float | pd.Series = 0) -> float | pd.Series | pd.DataFrame:
    # series & series -> float
    # series & benchmark(, m) -> series(m,)
    # fund(, n) & series -> dataframe(, n)
    # fund(, n) & benchmark(, m) -> dataframe(n, m)
    return (s).aggregate(lambda y: sm.OLS(y - rfr_periodic, sm.add_constant(benchmark)).fit().params).iloc[1:].squeeze()


@requireReturn
@requireBenchmark
def alpha(s: pd.Series | pd.DataFrame, benchmark: pd.Series | pd.DataFrame, rfr_periodic: float | pd.Series = 0, annualize=False) -> float | pd.Series | pd.DataFrame:
    a = (s).aggregate(lambda y: sm.OLS(y - rfr_periodic, sm.add_constant(benchmark)).fit().params).iloc[0].squeeze()
    if annualize:
        # Sharpe's definition
        a *= periodicity(s)
    return a


@requireReturn
@requireBenchmark
# FIXME: add test case
def rsquared(s: pd.Series | pd.DataFrame, benchmark: pd.Series | pd.DataFrame, rfr_periodic: float | pd.Series = 0, adjusted: bool = False) -> float | pd.Series | pd.DataFrame:
    result = (s).aggregate(lambda y: sm.OLS(
        y - rfr_periodic, sm.add_constant(benchmark)).fit())
    r2 = result.rsquared_adj if adjusted else result.rsquared
    return r2.squeeze()


@requireReturn
@requireBenchmark
# Unlike pure beta, it doens't make sense to calcualte bull/bear beta on multiple indices, so benchmark must not be a DataFrame
def bull_beta(s: pd.Series | pd.DataFrame, benchmark: pd.Series, rfr_periodic: float | pd.Series = 0) -> float | pd.Series | pd.DataFrame:
    bull = benchmark > rfr_periodic
    return beta(s[bull] - rfr_periodic[bull], benchmark[bull] - rfr_periodic[bull])


@requireReturn
@requireBenchmark
def bear_beta(s: pd.Series | pd.DataFrame, benchmark: pd.Series, rfr_periodic: float | pd.Series = 0) -> float | pd.Series | pd.DataFrame:
    bear = benchmark < rfr_periodic
    return beta(s[bear] - rfr_periodic[bear], benchmark[bear] - rfr_periodic[bear])


def beta_timing_ratio(s: pd.Series | pd.DataFrame, benchmark: pd.Series | pd.DataFrame, rfr_periodic: float | pd.Series = 0) -> float | pd.Series | pd.DataFrame:
    return bull_beta(s, benchmark, rfr_periodic) / bear_beta(s, benchmark, rfr_periodic)

# Only one benchmark


def treynor(s: pd.Series | pd.DataFrame, benchmark: pd.Series, rfr_periodic: float | pd.Series = 0, annualize=True) -> float | pd.Series:
    return (compound_return(s, annualize) - rfr_periodic) / beta(s, benchmark, rfr_periodic)


@requireReturn
@requireBenchmark
def tracking_error(s: pd.Series | pd.DataFrame, benchmark: pd.Series | pd.DataFrame, annualize: bool = False) -> float | pd.Series:
    return volatility(s - benchmark, annualize)


@requireReturn
@requireBenchmark
def information_ratio(s: pd.Series | pd.DataFrame, benchmark: pd.Series | pd.DataFrame) -> float | pd.Series:
    # Arithmetic
    return (compound_return(s, True) - compound_return(benchmark, True)) / tracking_error(s, benchmark, True)


@requireReturn
def summary(s: pd.Series | pd.DataFrame, benchmark: pd.Series = None, rfr_annualized: float = 0, mar: float = 0):

    rfr_periodic = (1 + rfr_annualized) ** (1 / periodicity(s)) - 1

    sd = {'Number of period': len(s),
         'Frequency': s.index.freqstr,
         'Total Return': compound_return(s),
         'Periodic mean return': arithmetic_mean(s),
         'Periodic geometric mean': geometric_mean(s),
         'Annualized return': compound_return(s, annualize=True),
         'Best period': best_period(s),
         'Worst period': worst_period(s),
         'Average positive period': avg_pos(s),
         'Average negative period': avg_neg(s),
         'Mean absolute deviation': mean_abs_dev(s),
         'Variance': variance(s),
         'Period Volatility': volatility(s),
         'Period volatility of positive return': vol_pos(s),
         'Period volatility of negative return': vol_neg(s),
         'Annualized Volatility': volatility(s, annualize=True),
         f'Sharpe ({rfr_annualized:.2%})': sharpe(s, rfr_annualized),
         'Skewness': skew(s),
         'Excess Kurtosis': kurt(s),
         'Normal (1%)': is_normal(s, 0.01),
         'VaR Historical (95%)': var_historical(s),
         'VaR Normal (95%)': var_normal(s),
         'VaR Modified (95%)': var_modified(s),

         # Drawdown
         'Worst Drawdown': worst_drawdown(s),
         'Calmar': calmar(s, rfr_annualized=rfr_annualized),
         'Average Drawdown': avg_drawdown(s, d=3),
         # ftk.sterling() # Original
         'Modified Sterling': sterling_modified(s, rfr_annualized=rfr_annualized, d=3),
         'Drawdown Deviation': drawdown_deviation(s, d=3),
         'Modified Burke': burke_modified(s, rfr_annualized=rfr_annualized, d=3),
         'Average Annual Drawdown': avg_annual_drawdown(s),
         'Sterling-Calmar': sterling_calmar(s, rfr_annualized=rfr_annualized, d=3),
         'Pain Index': pain_index(s),
         'Pain Ratio': pain(s, rfr_annualized),
         'Ulcer Index': ulcer_index(s),
         'Martin Ratio': martin(s, rfr_annualized),

         # Partial
         'Downside Potential': downside_potential(s, mar=mar),
         'Downside Risk (Periodic)': downside_risk(s, mar=mar),
         'Downside Risk (Annualized)': downside_risk(s, mar=mar, annualize=True),
         'Upside Potential': upside_potential(s, mar=mar),
         'Upside Risk (Periodic)': upside_risk(s, mar=mar),
         'Upside Risk (Annualized)': upside_risk(s, mar=mar, annualize=True),
         'Omega Ratio': omega(s, mar=mar),
         'Upside Potential Ratio': upside_potential_ratio(s, mar),
         'Variability Skewness': variability_skewness(s, mar),
         'Sortino Ratio': sortino(s, mar=mar)
         }
    if benchmark is not None:
        sd.update({
            # Relative
            'Tracking Error': tracking_error(s, benchmark),
            'Annualized Tracking Error': tracking_error(s, benchmark, True),
            'Annualized Information Ratio': information_ratio(s, benchmark),

            # Regression
            'Beta': beta(s, benchmark, rfr_periodic),
            'Alpha (Annualized)': alpha(s, benchmark, rfr_periodic, annualize=True),
            'Bull Beta': bull_beta(s, benchmark, rfr_periodic),
            'Bear Beta': bear_beta(s, benchmark, rfr_periodic),
            'Beta Timing Ratio': beta_timing_ratio(s, benchmark, rfr_periodic),
        })
    return sd

# TODO:
# Accept both 5% or 95%

@requireReturn
def is_normal(s: pd.Series, z: float = 0.01):
    # p-value > z means null hypothesis (normal) cannot be rejected
    return scipy.stats.jarque_bera(s)[1] > z

@requireReturn
def var_historical(s: pd.Series, z: float = 0.05) -> float:
    """_summary_

    Parameters
    ----------
    s : pd.Series
        _description_
    z : float, optional
        _description_, by default 0.05

    Returns
    -------
    float
        VaR, reported as a negative number
    """
    return np.percentile(s, z * 100)


def var_normal(s: pd.Series, a: float = 0.95) -> float:
    """Gaussian Value-at-Risk

    Parameters
    ----------
    s : pd.Series
        _description_
    a : float, optional
        _description_, by default 0.95

    Returns
    -------
    float
        VaR, reported as a negative number
    """
    z = -abs(scipy.stats.norm.ppf(a))
    mu = arithmetic_mean(s)
    sigma = volatility(s, annualize=False)
    return mu + sigma * z


def var_modified(s: pd.Series, a: float = 0.95) -> float:
    """Modified Value-at-Risk

    Parameters
    ----------
    s : pd.Series
        _description_
    a : float, optional
        _description_, by default 0.95

    Returns
    -------
    float
        mVaR, reported as a negative number
    """
    z = -abs(scipy.stats.norm.ppf(a))
    mu = arithmetic_mean(s)
    sigma = volatility(s, annualize=False)
    S = skew(s)
    K = kurt(s)
    t = z + (z ** 2 - 1) * S / 6 \
        + (z ** 3 - 3 * z) * K / 24 \
        - (2 * z ** 3 - 5 * z) * S ** 2 / 36
    print('Modified', mu, sigma, t, z, (z ** 2 - 1) * S / 6, (z ** 3 - 3 * z) * K / 24, - (2 * z ** 3 - 5 * z) * S ** 2 / 36)
    return mu + sigma * t


# To check
def cvar_historical(s: pd.Series, z: float = 0.01):
    return s[s < var_historical(s, z)].mean()

# To check
def cvar_normal(s: pd.Series, z: float = 0.01, annualize=False):
    mu = compound_return(s, annualize)
    sigma = volatility(s, annualize)
    return mu - sigma * scipy.stats.norm.pdf(scipy.stats.norm.ppf(z)) / z


@requireReturn
def up_capture():
    bm = requireBenchmarkReturn(benchmark)
    up = bm >= 0
    return compound_return(s[up], True) / compound(bm[up], True)


@requireReturn
def down_capture():
    bm = requireBenchmarkReturn(benchmark)
    down = bm < 0
    return compound_return(s[down], True) / compound(bm[down], True)


def ytd():
    pass


def mtd():
    pass


def drawdowns():
    pass


def m2():
    pass


def carino(r, b):
    return np.where(r == b, 1 / (1 + r), (np.log1p(r) - np.log1p(b)) / (r - b))
