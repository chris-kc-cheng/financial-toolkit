"""Most frequently used formulas in quantitative finance.

Functions are all vectorized and work for both Series and DataFrame
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
    return s.iloc[1:] #.dropna()


def return_to_price(r: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    if isinstance(r, pd.Series):
        s = pd.concat([pd.Series([0]), r])
    else:
        s = pd.concat(
            [pd.DataFrame(np.zeros((1, r.shape[1])), columns=r.columns), r])
    s.index = pd.date_range(
        end=r.index[-1].to_timestamp(how='e').date(), periods=len(r.index) + 1, freq=r.index.freq)
    return (s + 1).cumprod()

##############
# Decorators #
##############


def requireReturn(func):
    """Decorator that convert prices to returns if Index is DatetimeIndex

    Parameters
    ----------
    func : _type_
        _description_

    Returns
    -------
    _type_
        _description_
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
    """Decorator that convert returns to prices if Index is PeriodIndex

    Parameters
    ----------
    func : _type_
        _description_

    Returns
    -------
    _type_
        _description_
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
    """Decorator that convert benchmark values to returns if Index is
    DatetimeIndex

    Parameters
    ----------
    func : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    @wraps(func)
    def wrapper(x, pre, *args, **kwargs):
        post = pre
        if isinstance(pre.index, pd.DatetimeIndex):
            post = price_to_return(pre)
        return func(x, post, *args, **kwargs)
    return wrapper


def convertFX(s: pd.Series | pd.DataFrame, foreign: pd.Series, domestic: pd.Series) -> pd.Series | pd.DataFrame:
    """Convert the time series from foreign currency to domestic currency.

    Note
    ----
    Yahoo always use direct quote.
    (vs Bloomberg use indirect quote for GBP/EUR/AUD/NZD)

    Parameters
    ----------
    s : pd.Series | pd.DataFrame
        Time series of asset(s). It can be a Series or DataFrame of either
        prices or returns, but not a mix of both.
    foreign : pd.Series
        Foreign currency, which the currency that the asset is denominated
    domestic : pd.Series
        Domestic currency, which is the currency of the investor's home country

    Returns
    -------
    pd.Series | pd.DataFrame
        Time series of the prices
    """
    if type(s.index) == type(foreign.index) and type(s.index) == type(domestic.index) and periodicity(s) <= periodicity(foreign) and periodicity(s) <= periodicity(domestic):
        if isinstance(s.index, pd.DatetimeIndex):
            # Price
            # If no FX is quoted on a particular day, filled with the last quoted price
            _fc = foreign.reindex(s.index).ffill()
            _lc = domestic.reindex(s.index).ffill()
            return s.div(_fc, axis=0).mul(_lc, axis=0)
        else:
            # Return
            return convertFX(return_to_price(s), return_to_price(foreign), return_to_price(domestic))
    else:
        # Error converting, returning original
        return s


@requireReturn
def compound_return(ts: pd.Series | pd.DataFrame, annualize=False) -> float | pd.Series:
    """Compound return of time series

    Parameters
    ----------
    ts : pd.Series | pd.DataFrame
        Pandas Series or DataFrame of returns
    annualize : bool, optional
        Annualizing the compound return, by default False

    Returns
    -------
    float | pd.Series
        _description_
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
    # Drawdown must be calculated for each series individually
    if isinstance(p, pd.DataFrame):
        return p.aggregate(all_drawdown)
    m = p.cummax()
    peak = (p == m) & (p < m).shift(-1)
    num = peak.cumsum()
    dd = p.groupby(num).aggregate(worst_drawdown)
    return dd[dd < 0].sort_values()


@requireReturn
def avg_annual_drawdown(p: pd.Series | pd.DataFrame) -> float | pd.Series:
    return p.groupby(p.index.year).aggregate(worst_drawdown).mean()


def avg_drawdown(p: pd.Series | pd.DataFrame, d: int = 3) -> float | pd.Series:
    # Average drawdown must be calculated for each series individually
    if isinstance(p, pd.DataFrame):
        return p.aggregate(lambda s: avg_drawdown(s, 3))
    return all_drawdown(p)[:d].mean()


def calmar(s: pd.Series | pd.DataFrame, rfr_annualized: float = 0) -> float | pd.Series:
    return (compound_return(s, True) - rfr_annualized) / -worst_drawdown(s)


def sterling(s: pd.Series | pd.DataFrame) -> float | pd.Series:
    return compound_return(s, True) / np.absolute(avg_annual_drawdown(s) - 0.1)


def sterling_modified(s: pd.Series | pd.DataFrame, rfr_annualized: float = 0, d: int = 3) -> float | pd.Series:
    return (compound_return(s, True) - rfr_annualized) / np.absolute(avg_drawdown(s, d))


def sterling_calmar(s: pd.Series | pd.DataFrame, rfr_annualized: float = 0) -> float | pd.Series:
    return (compound_return(s, True) - rfr_annualized) / -avg_annual_drawdown(s)

@requireReturn
def drawdown_deviation(s: pd.Series | pd.DataFrame, d: int = 3) -> float | pd.Series:
    """Drawdown deviation measures the standard deviation of individual
    drawdowns. This is used the denominator of the Modified Burke ratio.

    Note: This is not the same as downside deviation.

    Parameters
    ----------
    s : pd.Series | pd.DataFrame
        _description_
    d : int, optional
        _description_, by default 3

    Returns
    -------
    float | pd.Series
        _description_
    """
    return -np.sqrt((all_drawdown(s)[:d] ** 2).sum() / (len(s)))


def burke_modified(s: pd.Series | pd.DataFrame, rfr_annualized: float = 0, d: int = 3) -> float | pd.Series:
    """Modified Burke ratio is a Sharpe-like ratio but uses drawdown deviation
    in the denominator.

    Parameters
    ----------
    s : pd.Series | pd.DataFrame
        _description_
    rfr_annualized : float, optional
        _description_, by default 0
    d : int, optional
        _description_, by default 3

    Returns
    -------
    float | pd.Series
        _description_
    """
    return (compound_return(s, True) - rfr_annualized) / -drawdown_deviation(s, d)


@requirePrice
def underwater(s: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return s / s.cummax() - 1


@requireReturn
# Positive
def ulcer_index(s: pd.Series | pd.DataFrame) -> float | pd.Series:
    """The ulcer index is similar to drawdown deviation but also take into
    account the time being underwater.

    Parameters
    ----------
    s : pd.Series | pd.DataFrame
        _description_

    Returns
    -------
    float | pd.Series
        _description_
    """
    return np.sqrt((underwater(s) ** 2).sum() / len(s))


@requireReturn
# Positive
def pain_index(s: pd.Series | pd.DataFrame) -> float | pd.Series:
    """The pain index is similar to the ulcer index but use absolute value of
    the underwater instead of squaring.

    Parameters
    ----------
    s : pd.Series | pd.DataFrame
        _description_

    Returns
    -------
    float | pd.Series
        _description_
    """
    return np.absolute(underwater(s)).sum() / len(s)


def martin(s: pd.Series | pd.DataFrame, rfr_annualized: float = 0) -> float | pd.Series:
    """Martin ratio is a Sharpe-like ratio but uses the ulcer index in the
    denominator.

    Also know as ulcer performance index.

    Parameters
    ----------
    s : pd.Series | pd.DataFrame
        _description_
    rfr_annualized : float, optional
        _description_, by default 0

    Returns
    -------
    float | pd.Series
        _description_
    """
    return (compound_return(s, True) - rfr_annualized) / ulcer_index(s)


def pain(s: pd.Series | pd.DataFrame, rfr_annualized: float = 0) -> float | pd.Series:
    """The pain ratio is a Sharpe-like ratio but uses the pain index in the
    denominator.

    Parameters
    ----------
    s : pd.Series | pd.DataFrame
        _description_
    rfr_annualized : float, optional
        _description_, by default 0

    Returns
    -------
    float | pd.Series
        _description_
    """
    return (compound_return(s, True) - rfr_annualized) / pain_index(s)


@requireReturn
def downside_potential(s: pd.Series | pd.DataFrame, mar: float = 0) -> float | pd.Series:
    return (mar - s[s < mar]).sum() / len(s)


@requireReturn
def upside_potential(s: pd.Series | pd.DataFrame, mar: float = 0) -> float | pd.Series:
    return (s[s > mar] - mar).sum() / len(s)


@requireReturn
def downside_risk(s: pd.Series | pd.DataFrame, mar: float = 0, annualize: bool = False, ddof: int = 0) -> float | pd.Series:
    """Downside Risk measures the variability of underperformance below a
    minimum target return. It is the denominator of a Sortino ratio.

    Also know as Downside Deviation.


    Parameters
    ----------
    s : pd.Series | pd.DataFrame
        _description_
    mar : float, optional
        _description_, by default 0
    annualize : bool, optional
        _description_, by default False
    ddof: int, optional
        Delta Degrees of Freedom, by default 0

    Returns
    -------
    float | pd.Series
        _description_
    """
    dr = np.sqrt(((mar - s[s < mar]) ** 2).sum() / (len(s) - ddof))
    if annualize:
        dr *= np.sqrt(periodicity(s))
    return dr


@requireReturn
def upside_risk(s: pd.Series | pd.DataFrame, mar: float = 0, annualize: bool = False, ddof: int = 0) -> float | pd.Series:
    """_summary_

    Parameters
    ----------
    s : pd.Series | pd.DataFrame
        _description_
    mar : float, optional
        _description_, by default 0
    annualize : bool, optional
        _description_, by default False
    ddof : int, optional
        Delta Degrees of Freedom, by default 0

    Returns
    -------
    float | pd.Series
        _description_
    """
    ur = np.sqrt(((s[s > mar] - mar) ** 2).sum() / len(s))
    if annualize:
        ur *= np.sqrt(periodicity(s))
    return ur


def omega(s: pd.Series | pd.DataFrame, mar: float = 0) -> float:
    """Omega ratio measures the gain-loss ratio, i.e. Upside potential divided
    by Downside potential.

    Parameters
    ----------
    s : pd.Series | pd.DataFrame
        _description_
    mar : float, optional
        _description_, by default 0

    Returns
    -------
    float
        _description_
    """
    return upside_potential(s, mar) / downside_potential(s, mar)


def sortino(s: pd.Series | pd.DataFrame, mar: float = 0, ddof: int = 0) -> float | pd.Series:
    ann_mar = (1 + mar) ** periodicity(s) - 1
    return (compound_return(s, annualize=True) - ann_mar) / downside_risk(s, mar, annualize=True, ddof=ddof)


def upside_potential_ratio(s: pd.Series | pd.DataFrame, mar: float = 0) -> float | pd.Series:
    return upside_potential(s, mar) / downside_risk(s, mar, annualize=True)


def variability_skewness(s: pd.Series | pd.DataFrame, mar: float = 0) -> float | pd.Series:
    """Variability skewness is the ratio of Upside risk compared to Downside
    risk.

    Parameters
    ----------
    s : pd.Series | pd.DataFrame
        _description_
    mar : float, optional
        _description_, by default 0

    Returns
    -------
    float | pd.Series
        _description_
    """
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
# f, b -> Series of shape (3,) (float, float, float)
# f, bb -> Series of shape (3,) (float, Series, float)
# ff, b -> DataFrame of shape (3, n)
# ff, bb -> DataFrame of shape (3, n), the middle row is Series
# where n is the number of assets (column) in s
def regress(s: pd.Series | pd.DataFrame, benchmark: pd.Series | pd.DataFrame, rfr_periodic: float | pd.Series = 0) -> pd.Series | pd.DataFrame:
    if isinstance(s, pd.DataFrame):
        return s.aggregate(lambda x: regress(x, benchmark, rfr_periodic))
    result = sm.OLS(s - rfr_periodic, sm.add_constant(benchmark)).fit()
    alpha = result.params.iloc[0]
    betas = result.params.iloc[1:].squeeze() # Series
    r2 = result.rsquared
    r2adj = result.rsquared_adj
    return pd.Series([alpha, betas, r2, r2adj], index=['alpha', 'betas', 'r2', 'r2adj'])


@requireReturn
@requireBenchmark
def beta(s: pd.Series | pd.DataFrame, benchmark: pd.Series | pd.DataFrame, rfr_periodic: float | pd.Series = 0) -> float | pd.Series | pd.DataFrame:
    # series & series -> float
    # series & benchmark(, m) -> Series(m,)
    # fund(, k) & series -> Series(, k)
    # fund(, k) & benchmark(, m) -> Series(k,) with the 2nd row being another Series(m,)
    # Note: benchmark should be already NET of risk-free rate, so this method works for multi-factor analysis
    return regress(s, benchmark, rfr_periodic).iloc[1]


@requireReturn
@requireBenchmark
def alpha(s: pd.Series | pd.DataFrame, benchmark: pd.Series | pd.DataFrame, rfr_periodic: float | pd.Series = 0, annualize=False) -> float | pd.Series | pd.DataFrame:
    a = regress(s, benchmark, rfr_periodic).iloc[0]
    if annualize:
        # Sharpe's definition
        a *= periodicity(s)
    return a


@requireReturn
@requireBenchmark
# FIXME: add test case
def rsquared(s: pd.Series | pd.DataFrame, benchmark: pd.Series | pd.DataFrame, rfr_periodic: float | pd.Series = 0, adjusted: bool = False) -> float | pd.Series | pd.DataFrame:
    result = regress(s, benchmark, rfr_periodic)
    return result.iloc[3] if adjusted else result.iloc[2]


@requireReturn
@requireBenchmark
# Unlike pure beta, it doens't make sense to calcualte bull/bear beta on multiple indices, so benchmark must not be a DataFrame
def bull_beta(s: pd.Series | pd.DataFrame, benchmark: pd.Series, rfr_periodic: float | pd.Series = 0) -> float | pd.Series | pd.DataFrame:
    bull = benchmark > rfr_periodic
    return beta(s[bull].subtract(rfr_periodic[bull], axis=0), benchmark[bull].subtract(rfr_periodic[bull], axis=0))


@requireReturn
@requireBenchmark
def bear_beta(s: pd.Series | pd.DataFrame, benchmark: pd.Series, rfr_periodic: float | pd.Series = 0) -> float | pd.Series | pd.DataFrame:
    bear = benchmark < rfr_periodic
    return beta(s[bear].subtract(rfr_periodic[bear], axis=0), benchmark[bear].subtract(rfr_periodic[bear], axis=0))


def beta_timing_ratio(s: pd.Series | pd.DataFrame, benchmark: pd.Series | pd.DataFrame, rfr_periodic: float | pd.Series = 0) -> float | pd.Series | pd.DataFrame:
    return bull_beta(s, benchmark, rfr_periodic) / bear_beta(s, benchmark, rfr_periodic)

# Only one benchmark


def treynor(s: pd.Series | pd.DataFrame, benchmark: pd.Series, rfr_periodic: float | pd.Series = 0, annualize=True) -> float | pd.Series:
    rfr_annualized = compound_return(rfr_periodic, annualize=True)
    return (compound_return(s, annualize) - rfr_annualized) / beta(s, benchmark, rfr_periodic)


@requireReturn
@requireBenchmark
def tracking_error(s: pd.Series | pd.DataFrame, benchmark: pd.Series | pd.DataFrame, annualize: bool = False) -> float | pd.Series:
    return volatility(s.subtract(benchmark, axis=0), annualize)

@requireReturn
@requireBenchmark
def active_return(s: pd.Series | pd.DataFrame, benchmark: pd.Series | pd.DataFrame, annualize=True) -> float | pd.Series:
    # Arithmetic
    return compound_return(s, True) - compound_return(benchmark, True)

@requireReturn
@requireBenchmark
def information_ratio(s: pd.Series | pd.DataFrame, benchmark: pd.Series | pd.DataFrame) -> float | pd.Series:
    # Arithmetic
    return active_return(s, benchmark, True) / tracking_error(s, benchmark, True)


@requireReturn
@requireBenchmark
def summary(s: pd.Series | pd.DataFrame, benchmark: pd.Series, rfr_periodic, mar: float = 0):

    rfr_annualized = compound_return(rfr_periodic, annualize=True)
    s_rfr = s.subtract(rfr_periodic, axis=0)
    mkt_rfr = benchmark - rfr_periodic

    sd = {
            'Number of Period': len(s),
            'Frequency': s.index.freqstr,
            'Total Return': compound_return(s),
            'Periodic Mean Return': arithmetic_mean(s),
            'Periodic Geometric Mean': geometric_mean(s),
            'Annualized Return': compound_return(s, annualize=True),
            'Best Period': best_period(s),
            'Worst Period': worst_period(s),
            'Average Positive Period': avg_pos(s),
            'Average Negative Period': avg_neg(s),
            'Mean Absolute Deviation': mean_abs_dev(s),
            'Variance': variance(s),
            'Period Volatility': volatility(s),
            'Period Volatility of Positive Return': vol_pos(s),
            'Period Volatility of Negative Return': vol_neg(s),
            'Annualized Volatility': volatility(s, annualize=True),
            f'Sharpe ({rfr_annualized:.2%})': sharpe(s, rfr_annualized),
            'Skewness': skew(s),
            'Excess Kurtosis': kurt(s),
            # The following are not tested
            'Normal (1%)': is_normal(s, 0.01),
            'VaR Historical (95%)': var_historical(s),
            'VaR Gaussian (95%)': var_normal(s),
            'VaR Modified (95%)': var_modified(s),
            'CVaR Historical (95%)': cvar_historical(s),
            'CVaR Gaussian (95%)': cvar_normal(s),

            # Drawdown
            'Worst Drawdown': worst_drawdown(s),
            'Calmar': calmar(s, rfr_annualized=rfr_annualized),
            'Average Drawdown': avg_drawdown(s, d=3),
            'Sterling Original': sterling(s),  # Not tested
            'Sterling Modified': sterling_modified(s, rfr_annualized=rfr_annualized, d=3),
            'Sterling-Calmar': sterling_calmar(s, rfr_annualized=rfr_annualized),
            'Drawdown Deviation': drawdown_deviation(s, d=3),
            'Modified Burke': burke_modified(s, rfr_annualized=rfr_annualized, d=3),
            'Average Annual Drawdown': avg_annual_drawdown(s),          
            'Pain Index': pain_index(s),
            'Pain Ratio': pain(s, rfr_annualized),
            'Ulcer Index': ulcer_index(s),
            'Martin Ratio': martin(s, rfr_annualized),

            # Partial
            'Downside Potential': downside_potential(s, mar=mar),
            'Downside Risk (Periodic)': downside_risk(s, mar=mar, ddof=1),
            'Downside Risk (Annualized)': downside_risk(s, mar=mar, annualize=True, ddof=1),
            'Upside Potential': upside_potential(s, mar=mar),
            'Upside Risk (Periodic)': upside_risk(s, mar=mar, ddof=1),
            'Upside Risk (Annualized)': upside_risk(s, mar=mar, annualize=True, ddof=1),
            'Omega Ratio': omega(s, mar=mar),
            'Upside Potential Ratio': upside_potential_ratio(s, mar),
            'Variability Skewness': variability_skewness(s, mar),
            'Sortino Ratio': sortino(s, mar=mar, ddof=1),

            # Relative
            'Tracking Error': tracking_error(s, benchmark),
            'Annualized Tracking Error': tracking_error(s, benchmark, True),
            'Annualized Active Return': active_return(s, benchmark, True),
            'Annualized Information Ratio': information_ratio(s, benchmark),

            # Regression
            'Beta': beta(s, mkt_rfr, rfr_periodic),
            'Alpha (Annualized)': alpha(s, mkt_rfr, rfr_periodic, annualize=True),
            'Correlation': correlation(pd.concat([s_rfr, mkt_rfr], axis=1)).iloc[-1, :-1].squeeze(),
            'R-Squared': rsquared(s, mkt_rfr, rfr_periodic),
            'Bull Beta': bull_beta(s, benchmark, rfr_periodic), # Bull/Bear/Timing not tested
            'Bear Beta': bear_beta(s, benchmark, rfr_periodic),
            'Beta Timing Ratio': beta_timing_ratio(s, benchmark, rfr_periodic),
            'Treynor Ratio': treynor(s, benchmark, rfr_periodic),
            'Up Capture': up_capture(s, benchmark),
            'Down Capture': down_capture(s, benchmark)
        }
    return sd

# TODO:
# Accept both 5% (Level of significance) or 95% (Confidence Interval)


@requireReturn
def is_normal(s: pd.Series, a: float = 0.01):
    # p-value > z means null hypothesis (normal) cannot be rejected
    return scipy.stats.jarque_bera(s)[1] > a


@requireReturn
def var_historical(s: pd.Series, alpha: float = 0.95) -> float:
    """Historical Value-at-Risk

    Parameters
    ----------
    s : pd.Series
        _description_
    alpha : float, optional
        _description_, by default 0.05

    Returns
    -------
    float
        VaR, reported as a negative number
    """
    a = min(alpha, 1 - alpha)
    return s.quantile(a)


def var_normal(s: pd.Series, alpha: float = 0.95) -> float:
    """Gaussian Value-at-Risk

    Parameters
    ----------
    s : pd.Series
        _description_
    alpha : float, optional
        _description_, by default 0.95

    Returns
    -------
    float
        VaR, reported as a negative number
    """
    z = -abs(scipy.stats.norm.ppf(alpha))
    mu = arithmetic_mean(s)
    sigma = volatility(s, annualize=False)
    return mu + sigma * z


def var_modified(s: pd.Series, alpha: float = 0.95) -> float:
    """Modified Value-at-Risk

    Parameters
    ----------
    s : pd.Series
        _description_
    alpha : float, optional
        _description_, by default 0.95

    Returns
    -------
    float
        mVaR, reported as a negative number
    """
    z = -abs(scipy.stats.norm.ppf(alpha))
    mu = arithmetic_mean(s)
    sigma = volatility(s, annualize=False)
    S = skew(s)
    K = kurt(s)
    t = z + (z ** 2 - 1) * S / 6 \
        + (z ** 3 - 3 * z) * K / 24 \
        - (2 * z ** 3 - 5 * z) * S ** 2 / 36
    return mu + sigma * t


# TODO: Add test case
@requireReturn
def cvar_historical(s: pd.Series, alpha: float = 0.05):
    """Historical Conditional Value-at-Risk (CVaR).

    Also known as Expected Shortfall (ES).

    Parameters
    ----------
    s : pd.Series
        _description_
    alpha : float, optional
        _description_, by default 0.01

    Returns
    -------
    _type_
        CVaR, reported as a negative number
    """
    return s[s < var_historical(s, alpha)].mean()

# TODO: Add test case


def cvar_normal(s: pd.Series, alpha: float = 0.95, annualize=False):
    """Gaussian Conditional Value-at-Risk (CVaR).

    Also known as Expected Shortfall (ES).

    Parameters
    ----------
    s : pd.Series
        _description_
    alpha : float, optional
        _description_, by default 0.95
    annualize : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        CVaR, reported as a negative number
    """
    a = min(alpha, 1 - alpha)
    mu = arithmetic_mean(s)
    sigma = volatility(s, annualize)
    return mu - sigma * scipy.stats.norm.pdf(scipy.stats.norm.ppf(a)) / a


@requireReturn
@requireBenchmark
def up_capture(s: pd.Series | pd.DataFrame, benchmark: pd.Series | pd.DataFrame):
    up = benchmark >= 0
    return compound_return(s[up]) / compound_return(benchmark[up])


@requireReturn
@requireBenchmark
def down_capture(s: pd.Series | pd.DataFrame, benchmark: pd.Series | pd.DataFrame):
    down = benchmark < 0
    return compound_return(s[down]) / compound_return(benchmark[down])


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
