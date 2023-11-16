"""Option Greeks

All functions share the same signature and are vectorized.
"""
import numpy as np
from scipy import stats

def bp(face: float | np.ndarray = 100., rate: float = 0.05, time: float = 0.25) -> float | np.ndarray:
    """Present value of a zero-coupon bond.
    
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


def d1(strike: float = 100., spot: float | np.ndarray = 100., rate: float = 0.05, time: float = 0.25, vol: float = 0.2, dvd: float = 0.) -> float | np.ndarray:
    """Probability of an European option being exercised.

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
    return (np.log(spot / strike) + (rate - dvd + vol ** 2 / 2) * time) / (vol * np.sqrt(time))

def d2(strike: float = 100., spot: float | np.ndarray = 100., rate: float = 0.05, time: float = 0.25, vol: float = 0.2, dvd: float = 0.) -> float | np.ndarray:
    """Probability of an European option **not** being exercised.

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

def price_call(strike: float = 100., spot: float | np.ndarray = 100., rate: float = 0.05, time: float = 0.25, vol: float = 0.2, dvd: float = 0.) -> float | np.ndarray:
    """Fair value of an European call option.

    .. math::
        Se^{-q \\tau}\Phi(d_1) - e^{-r \\tau} K\Phi(d_2)

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
    return spot * np.exp(-dvd * time) * stats.norm.cdf(d1(strike, spot, rate, time, vol, dvd))\
           - strike * np.exp(-rate * time) * stats.norm.cdf(d2(strike, spot, rate, time, vol, dvd))

def price_put(strike: float = 100., spot: float | np.ndarray = 100., rate: float = 0.05, time: float = 0.25, vol: float = 0.2, dvd: float = 0.) -> float | np.ndarray:
    """Fair value of an European put option.

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
    return strike * np.exp(-rate * time) * stats.norm.cdf(-d2(strike, spot, rate, time, vol, dvd))\
        - spot * np.exp(-dvd * time) * stats.norm.cdf(-d1(strike, spot, rate, time, vol, dvd))

def delta_call(strike: float = 100., spot: float | np.ndarray = 100., rate: float = 0.05, time: float = 0.25, vol: float = 0.2, dvd: float = 0.) -> float | np.ndarray:
    """Delta of an European call option.

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

def delta_put(strike: float = 100., spot: float | np.ndarray = 100., rate: float = 0.05, time: float = 0.25, vol: float = 0.2, dvd: float = 0.) -> float | np.ndarray:
    """Delta of an European put option.

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
    return -np.exp(-dvd * time) * stats.norm.cdf(-d1(strike, spot, rate, time, vol, dvd))

def vega(strike: float = 100., spot: float | np.ndarray = 100., rate: float = 0.05, time: float = 0.25, vol: float = 0.2, dvd: float = 0.) -> float | np.ndarray:
    """Vega of an European call/put option.

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
    return spot * np.exp(-dvd * time) * stats.norm.pdf(d1(strike, spot, rate, time, vol, dvd)) * np.sqrt(time)

def gamma(strike: float = 100., spot: float | np.ndarray = 100., rate: float = 0.05, time: float = 0.25, vol: float = 0.2, dvd: float = 0.) -> float | np.ndarray:
    """Gamma of an European call/put option.

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
    return np.exp(-dvd * time) * stats.norm.pdf(d1(strike, spot, rate, time, vol, dvd)) / spot / (vol * np.sqrt(time))

def theta_call(strike: float = 100., spot: float | np.ndarray = 100., rate: float = 0.05, time: float = 0.25, vol: float = 0.2, dvd: float = 0.) -> float | np.ndarray:
    """Theta of an European call option.

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
    return np.exp(-dvd * time) * -spot * stats.norm.pdf(d1(strike, spot, rate, time, vol, dvd)) * vol / 2 / np.sqrt(time)\
           - rate * strike * np.exp(-rate * time) * stats.norm.cdf(d2(strike, spot, rate, time, vol, dvd))\
           + dvd * spot * np.exp(-dvd * time) * stats.norm.cdf(d1(strike, spot, rate, time, vol, dvd))

def theta_put(strike: float = 100., spot: float | np.ndarray = 100., rate: float = 0.05, time: float = 0.25, vol: float = 0.2, dvd: float = 0.) -> float | np.ndarray:
    """Theta of an European put option.

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
    return np.exp(-dvd * time) * -spot * stats.norm.pdf(d1(strike, spot, rate, time, vol, dvd)) * vol / 2 / np.sqrt(time)\
           + rate * strike * np.exp(-rate * time) * stats.norm.cdf(-d2(strike, spot, rate, time, vol, dvd))\
           - dvd * spot * np.exp(-dvd * time) * stats.norm.cdf(-d1(strike, spot, rate, time, vol, dvd))

def rho_call(strike: float = 100., spot: float | np.ndarray = 100., rate: float = 0.05, time: float = 0.25, vol: float = 0.2, dvd: float = 0.) -> float | np.ndarray:
    """Rho of an European call option.

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
    return strike * time * np.exp(-rate * time) * stats.norm.cdf(d2(strike, spot, rate, time, vol, dvd))

def rho_put(strike: float = 100., spot: float | np.ndarray = 100., rate: float = 0.05, time: float = 0.25, vol: float = 0.2, dvd: float = 0.) -> float | np.ndarray:
    """Rho of an European put option.

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
    return -strike * time * np.exp(-rate * time) * stats.norm.cdf(-d2(strike, spot, rate, time, vol, dvd))
