"""Option Greeks

All functions share the same signature and are vectorized.
"""
import numpy as np
from scipy import stats


def bp(
    face: float | np.ndarray = 100.0, rate: float = 0.05, time: float = 0.25
) -> float | np.ndarray:
    """Present value of a zero-coupon bond, using continuous compounding.

    Auxiliary method to verify put-call parity hold.

    Parameters
    ----------
    face : float | np.ndarray
        Face value of the bond, by default 100.
    rate : float, optional
        Risk-free rate in decimal, by default 0.05 (i.e. 5%)
    time : float, optional
        Time to maturity, number of years in decimal, by default 0.25 (i.e. 3 months)

    Returns
    -------
    float | np.ndarray
        Present value of a zero-coupon bond.
    """
    return face * np.exp(-rate * time)


def d1(
    strike: float = 100.0,
    spot: float | np.ndarray = 100.0,
    rate: float = 0.05,
    time: float = 0.25,
    vol: float = 0.2,
    dvd: float = 0.0,
) -> float | np.ndarray:
    """Probability of an European option being exercised.

    .. math::
        d_1 = \\frac{\ln(S/K) + (r - q + \\frac{1}{2}\sigma^2)\\tau}{\sigma\sqrt{\\tau}}

    Parameters
    ----------
    strike : float, optional
        Strike price, by default 100.
    spot : float | np.ndarray, optional
        Stock price, by default 100.
    rate : float, optional
        Risk-free rate in decimal, by default 0.05 (i.e. 5%)
    time : float, optional
        Time to maturity, number of years in decimal, by default 0.25 (i.e. 3 months)
    vol : float, optional
        Volatility in decimal, by default 0.2 (i.e. 20%)
    dvd : float, optional
        Annual dividend yield in decimal, by default 0. (i.e. non-dividend paying)

    Returns
    -------
    float | np.ndarray
        Probability of an European option being exercised.
    """
    return (np.log(spot / strike) + (rate - dvd + vol**2 / 2) * time) / (
        vol * np.sqrt(time)
    )


def d2(
    strike: float = 100.0,
    spot: float | np.ndarray = 100.0,
    rate: float = 0.05,
    time: float = 0.25,
    vol: float = 0.2,
    dvd: float = 0.0,
) -> float | np.ndarray:
    """Probability of an European option **not** being exercised.

    .. math::
        d_2 = \\frac{\ln(S/K) + (r - q - \\frac{1}{2}\sigma^2)\\tau}{\sigma\sqrt{\\tau}} = d_1 - \sigma\sqrt{\\tau}

    Parameters
    ----------
    strike : float, optional
        Strike price, by default 100.
    spot : float | np.ndarray, optional
        Stock price, by default 100.
    rate : float, optional
        Risk-free rate in decimal, by default 0.05 (i.e. 5%)
    time : float, optional
        Time to maturity, number of years in decimal, by default 0.25 (i.e. 3 months)
    vol : float, optional
        Volatility in decimal, by default 0.2 (i.e. 20%)
    dvd : float, optional
        Annual dividend yield in decimal, by default 0. (i.e. non-dividend paying)

    Returns
    -------
    float | np.ndarray
        Probability of an European option not being exercised.
    """
    return d1(strike, spot, rate, time, vol, dvd) - vol * np.sqrt(time)


def price_call(
    strike: float = 100.0,
    spot: float | np.ndarray = 100.0,
    rate: float = 0.05,
    time: float = 0.25,
    vol: float = 0.2,
    dvd: float = 0.0,
) -> float | np.ndarray:
    """Fair value of an European call option.

    .. math::
        V_{call} = Se^{-q \\tau}\Phi(d_1) - e^{-r \\tau} K\Phi(d_2)

    Parameters
    ----------
    strike : float, optional
        Strike price, by default 100.
    spot : float | np.ndarray, optional
        Stock price, by default 100.
    rate : float, optional
        Risk-free rate in decimal, by default 0.05 (i.e. 5%)
    time : float, optional
        Time to maturity, number of years in decimal, by default 0.25 (i.e. 3 months)
    vol : float, optional
        Volatility in decimal, by default 0.2 (i.e. 20%)
    dvd : float, optional
        Annual dividend yield in decimal, by default 0. (i.e. non-dividend paying)

    Returns
    -------
    float | np.ndarray
        Fair value of the call option, in the same shape as spot.
    """
    return spot * np.exp(-dvd * time) * stats.norm.cdf(
        d1(strike, spot, rate, time, vol, dvd)
    ) - strike * np.exp(-rate * time) * stats.norm.cdf(
        d2(strike, spot, rate, time, vol, dvd)
    )


def price_put(
    strike: float = 100.0,
    spot: float | np.ndarray = 100.0,
    rate: float = 0.05,
    time: float = 0.25,
    vol: float = 0.2,
    dvd: float = 0.0,
) -> float | np.ndarray:
    """Fair value of an European put option.

    .. math::
        V_{put} = e^{-r \\tau} K\Phi(-d_2) -  Se^{-q \\tau}\Phi(-d_1)

    Parameters
    ----------
    strike : float, optional
        Strike price, by default 100.
    spot : float | np.ndarray, optional
        Stock price, by default 100.
    rate : float, optional
        Risk-free rate in decimal, by default 0.05 (i.e. 5%)
    time : float, optional
        Time to maturity, number of years in decimal, by default 0.25 (i.e. 3 months)
    vol : float, optional
        Volatility in decimal, by default 0.2 (i.e. 20%)
    dvd : float, optional
        Annual dividend yield in decimal, by default 0. (i.e. non-dividend paying)

    Returns
    -------
    float | np.ndarray
        Fair value of the put option, in the same shape as spot.
    """
    return strike * np.exp(-rate * time) * stats.norm.cdf(
        -d2(strike, spot, rate, time, vol, dvd)
    ) - spot * np.exp(-dvd * time) * stats.norm.cdf(
        -d1(strike, spot, rate, time, vol, dvd)
    )


def delta_call(
    strike: float = 100.0,
    spot: float | np.ndarray = 100.0,
    rate: float = 0.05,
    time: float = 0.25,
    vol: float = 0.2,
    dvd: float = 0.0,
) -> float | np.ndarray:
    """Delta of an European call option.

    Delta is the rate of change of the option price with respect to the price of the underlying asset.

    For a call option with delta of 0.5, its value will increase by $5 when the price of the underlying asset increases by $10.

    .. math::
        \Delta_{call} = e^{-q \\tau} \Phi(d_1)

    Parameters
    ----------
    strike : float, optional
        Strike price, by default 100.
    spot : float | np.ndarray, optional
        Stock price, by default 100.
    rate : float, optional
        Risk-free rate in decimal, by default 0.05 (i.e. 5%)
    time : float, optional
        Time to maturity, number of years in decimal, by default 0.25 (i.e. 3 months)
    vol : float, optional
        Volatility in decimal, by default 0.2 (i.e. 20%)
    dvd : float, optional
        Annual dividend yield in decimal, by default 0. (i.e. non-dividend paying)

    Returns
    -------
    float | np.ndarray
        Delta of the call option, in the same shape as spot.
    """
    return np.exp(-dvd * time) * stats.norm.cdf(d1(strike, spot, rate, time, vol, dvd))


def delta_put(
    strike: float = 100.0,
    spot: float | np.ndarray = 100.0,
    rate: float = 0.05,
    time: float = 0.25,
    vol: float = 0.2,
    dvd: float = 0.0,
) -> float | np.ndarray:
    """Delta of an European put option.

    .. math::
        \Delta_{put} = -e^{-q \\tau} \Phi(-d_1)

    Delta is the rate of change of the option price with respect to the price of the underlying asset.

    For a put option with delta of -0.5, its value will decrease by $5 when the price of the underlying asset increases by $10.

    Parameters
    ----------
    strike : float, optional
        Strike price, by default 100.
    spot : float | np.ndarray, optional
        Stock price, by default 100.
    rate : float, optional
        Risk-free rate in decimal, by default 0.05 (i.e. 5%)
    time : float, optional
        Time to maturity, number of years in decimal, by default 0.25 (i.e. 3 months)
    vol : float, optional
        Volatility in decimal, by default 0.2 (i.e. 20%)
    dvd : float, optional
        Annual dividend yield in decimal, by default 0. (i.e. non-dividend paying)

    Returns
    -------
    float | np.ndarray
        Delta of the put option, in the same shape as spot.
    """
    return -np.exp(-dvd * time) * stats.norm.cdf(
        -d1(strike, spot, rate, time, vol, dvd)
    )


def vega(
    strike: float = 100.0,
    spot: float | np.ndarray = 100.0,
    rate: float = 0.05,
    time: float = 0.25,
    vol: float = 0.2,
    dvd: float = 0.0,
) -> float | np.ndarray:
    """Vega of an European call/put option.

    .. math::
        \mathcal{V} = S e^{-q \\tau} \\varphi(d_1) \sqrt{\\tau} = K e^{-r \\tau} \\varphi(d_2) \sqrt{\\tau}

    Vega is the rate of change of the value of the portfolio with respect to the volatility of the underlying asset.
    For an option with vega of 15, its value will increase by $0.15 when the volatility of the underlying asset increase by 0.01 (e.g. from 20% to 21%).

    In practice, financial data providers often report vega as *vega per basis point*, which is the "textbook" vega / 100.
    In such case, the value of an option with vega of 0.15 will increase by $0.15 when the volatility of the underlying increase by 1%.

    Parameters
    ----------
    strike : float, optional
        Strike price, by default 100.
    spot : float | np.ndarray, optional
        Stock price, by default 100.
    rate : float, optional
        Risk-free rate in decimal, by default 0.05 (i.e. 5%)
    time : float, optional
        Time to maturity, number of years in decimal, by default 0.25 (i.e. 3 months)
    vol : float, optional
        Volatility in decimal, by default 0.2 (i.e. 20%)
    dvd : float, optional
        Annual dividend yield in decimal, by default 0. (i.e. non-dividend paying)

    Returns
    -------
    float | np.ndarray
        Vega of the option, in the same shape as spot.
    """
    return (
        spot
        * np.exp(-dvd * time)
        * stats.norm.pdf(d1(strike, spot, rate, time, vol, dvd))
        * np.sqrt(time)
    )


def gamma(
    strike: float = 100.0,
    spot: float | np.ndarray = 100.0,
    rate: float = 0.05,
    time: float = 0.25,
    vol: float = 0.2,
    dvd: float = 0.0,
) -> float | np.ndarray:
    """Gamma of an European call/put option.

    .. math::
        \Gamma = e^{-q \\tau} \\frac{\\varphi(d_1)}{S\sigma\sqrt{\\tau}} = K e^{-r \\tau} \\frac{\\varphi(d_2)}{S^2\sigma\sqrt{\\tau}}

    Gamma is the rate of change of the portfolio’s delta with respect to the price of the underlying asset.

    For an option with gamma of 0.1, its **delta** will increase by 0.5 when the price of the underlying asset increase by $5.
    The value of the option will approximately increase by $\\frac{1}{2}\Gamma \\times (\\delta S)^2$ which is $1.25.

    Parameters
    ----------
    strike : float, optional
        Strike price, by default 100.
    spot : float | np.ndarray, optional
        Stock price, by default 100.
    rate : float, optional
        Risk-free rate in decimal, by default 0.05 (i.e. 5%)
    time : float, optional
        Time to maturity, number of years in decimal, by default 0.25 (i.e. 3 months)
    vol : float, optional
        Volatility in decimal, by default 0.2 (i.e. 20%)
    dvd : float, optional
        Annual dividend yield in decimal, by default 0. (i.e. non-dividend paying)

    Returns
    -------
    float | np.ndarray
        Gamma of the option, in the same shape as spot.
    """
    return (
        np.exp(-dvd * time)
        * stats.norm.pdf(d1(strike, spot, rate, time, vol, dvd))
        / spot
        / (vol * np.sqrt(time))
    )


def theta_call(
    strike: float = 100.0,
    spot: float | np.ndarray = 100.0,
    rate: float = 0.05,
    time: float = 0.25,
    vol: float = 0.2,
    dvd: float = 0.0,
) -> float | np.ndarray:
    """Theta(call) of an European call option.

    .. math::
        \Theta_{call} = e^{-q \\tau} \\frac{S \\varphi(d_1) \sigma}{2 \sqrt{\\tau}} - rKe^{-r \\tau}\Phi(d_2) + qSe^{-q \\tau}\Phi(d_1)

    Theta is the rate of change of the value of the portfolio with respect to the passage of time.

    For an option with theta of -4, its value will decrease by $4 / 252 per trading day.

    Parameters
    ----------
    strike : float, optional
        Strike price, by default 100.
    spot : float | np.ndarray, optional
        Stock price, by default 100.
    rate : float, optional
        Risk-free rate in decimal, by default 0.05 (i.e. 5%)
    time : float, optional
        Time to maturity, number of years in decimal, by default 0.25 (i.e. 3 months)
    vol : float, optional
        Volatility in decimal, by default 0.2 (i.e. 20%)
    dvd : float, optional
        Annual dividend yield in decimal, by default 0. (i.e. non-dividend paying)

    Returns
    -------
    float | np.ndarray
        Theta of the call option, in the same shape as spot.
    """
    return (
        np.exp(-dvd * time)
        * -spot
        * stats.norm.pdf(d1(strike, spot, rate, time, vol, dvd))
        * vol
        / 2
        / np.sqrt(time)
        - rate
        * strike
        * np.exp(-rate * time)
        * stats.norm.cdf(d2(strike, spot, rate, time, vol, dvd))
        + dvd
        * spot
        * np.exp(-dvd * time)
        * stats.norm.cdf(d1(strike, spot, rate, time, vol, dvd))
    )


def theta_put(
    strike: float = 100.0,
    spot: float | np.ndarray = 100.0,
    rate: float = 0.05,
    time: float = 0.25,
    vol: float = 0.2,
    dvd: float = 0.0,
) -> float | np.ndarray:
    """Theta of an European put option.

    .. math::
        \Theta_{put} = e^{-q \\tau}\\frac{S \\varphi(d_1) \sigma}{2 \sqrt{\\tau}} + rKe^{-r \\tau}\Phi(-d_2) - qSe^{-q \\tau}\Phi(-d_1)

    Theta is the rate of change of the value of the portfolio with respect to the passage of time.

    For an option with theta of -4, its value will decrease by $4 / 252 per trading day.

    Parameters
    ----------
    strike : float, optional
        Strike price, by default 100.
    spot : float | np.ndarray, optional
        Stock price, by default 100.
    rate : float, optional
        Risk-free rate in decimal, by default 0.05 (i.e. 5%)
    time : float, optional
        Time to maturity, number of years in decimal, by default 0.25 (i.e. 3 months)
    vol : float, optional
        Volatility in decimal, by default 0.2 (i.e. 20%)
    dvd : float, optional
        Annual dividend yield in decimal, by default 0. (i.e. non-dividend paying)

    Returns
    -------
    float | np.ndarray
        Theta of the put option, in the same shape as spot.
    """
    return (
        np.exp(-dvd * time)
        * -spot
        * stats.norm.pdf(d1(strike, spot, rate, time, vol, dvd))
        * vol
        / 2
        / np.sqrt(time)
        + rate
        * strike
        * np.exp(-rate * time)
        * stats.norm.cdf(-d2(strike, spot, rate, time, vol, dvd))
        - dvd
        * spot
        * np.exp(-dvd * time)
        * stats.norm.cdf(-d1(strike, spot, rate, time, vol, dvd))
    )


def rho_call(
    strike: float = 100.0,
    spot: float | np.ndarray = 100.0,
    rate: float = 0.05,
    time: float = 0.25,
    vol: float = 0.2,
    dvd: float = 0.0,
) -> float | np.ndarray:
    """Rho of an European call option.

    .. math::
        \\rho_{call} = K \\tau e^{-r \\tau}\Phi(d_2)

    Parameters
    ----------
    strike : float, optional
        Strike price, by default 100.
    spot : float | np.ndarray, optional
        Stock price, by default 100.
    rate : float, optional
        Risk-free rate in decimal, by default 0.05 (i.e. 5%)
    time : float, optional
        Time to maturity, number of years in decimal, by default 0.25 (i.e. 3 months)
    vol : float, optional
        Volatility in decimal, by default 0.2 (i.e. 20%)
    dvd : float, optional
        Annual dividend yield in decimal, by default 0. (i.e. non-dividend paying)

    Returns
    -------
    float | np.ndarray
        Rho of the call option, in the same shape as spot.
    """
    return (
        strike
        * time
        * np.exp(-rate * time)
        * stats.norm.cdf(d2(strike, spot, rate, time, vol, dvd))
    )


def rho_put(
    strike: float = 100.0,
    spot: float | np.ndarray = 100.0,
    rate: float = 0.05,
    time: float = 0.25,
    vol: float = 0.2,
    dvd: float = 0.0,
) -> float | np.ndarray:
    """Rho of an European put option.

    .. math::
        \\rho_{put} = -K \\tau e^{-r \\tau}\Phi(-d_2)

    Parameters
    ----------
    strike : float, optional
        Strike price, by default 100.
    spot : float | np.ndarray, optional
        Stock price, by default 100.
    rate : float, optional
        Risk-free rate in decimal, by default 0.05 (i.e. 5%)
    time : float, optional
        Time to maturity, number of years in decimal, by default 0.25 (i.e. 3 months)
    vol : float, optional
        Volatility in decimal, by default 0.2 (i.e. 20%)
    dvd : float, optional
        Annual dividend yield in decimal, by default 0. (i.e. non-dividend paying)

    Returns
    -------
    float | np.ndarray
        Rho of the put option, in the same shape as spot.
    """
    return (
        -strike
        * time
        * np.exp(-rate * time)
        * stats.norm.cdf(-d2(strike, spot, rate, time, vol, dvd))
    )
