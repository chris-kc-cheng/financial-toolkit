"""Option Greeks
"""
import numpy as np
from scipy import stats

def bp(face, rate, time):
        return face * np.exp(-rate * time)

# All functions share the same signature
# Vectorized

def d1(strike, spot, rate, time, vol, dvd):
    return (np.log(spot / strike) + (rate - dvd + vol ** 2 / 2) * time) / (vol * np.sqrt(time))

def d2(strike, spot, rate, time, vol, dvd):
    return d1(strike, spot, rate, time, vol, dvd) - vol * np.sqrt(time)

def price_call(strike, spot, rate, time, vol, dvd):
    """Value of an European Call Option
    
    .. math:: Se^{-q \\tau}\Phi(d_1) - e^{-r \\tau} K\Phi(d_2)

    Args:
        strike (_type_): Exercise price
        spot (_type_): Spot price
        rate (_type_): _description_
        time (_type_): _description_
        vol (_type_): _description_
        dvd (_type_): _description_

    Returns:
        _type_: _description_
    """
    return spot * np.exp(-dvd * time) * stats.norm.cdf(d1(strike, spot, rate, time, vol, dvd))\
           - strike * np.exp(-rate * time) * stats.norm.cdf(d2(strike, spot, rate, time, vol, dvd))

def price_put(strike, spot, rate, time, vol, dvd):
    """Value of an European Put Option

    .. math::
        e^{-r \\tau} K\Phi(-d_2) -  Se^{-q \\tau}\Phi(-d_1)

    Args:
        strike (_type_): _description_
        spot (_type_): _description_
        rate (_type_): _description_
        time (_type_): _description_
        vol (_type_): _description_
        dvd (_type_): _description_

    Returns:
        _type_: _description_
    """
    return strike * np.exp(-rate * time) * stats.norm.cdf(-d2(strike, spot, rate, time, vol, dvd))\
        - spot * np.exp(-dvd * time) * stats.norm.cdf(-d1(strike, spot, rate, time, vol, dvd))

def delta_call(strike, spot, rate, time, vol, dvd):
    return np.exp(-dvd * time) * stats.norm.cdf(d1(strike, spot, rate, time, vol, dvd))

def delta_put(strike, spot, rate, time, vol, dvd):
    return -np.exp(-dvd * time) * stats.norm.cdf(-d1(strike, spot, rate, time, vol, dvd))

def vega(strike, spot, rate, time, vol, dvd):
    return spot * np.exp(-dvd * time) * stats.norm.pdf(d1(strike, spot, rate, time, vol, dvd)) * np.sqrt(time)

def gamma(strike, spot, rate, time, vol, dvd):
    return np.exp(-dvd * time) * stats.norm.pdf(d1(strike, spot, rate, time, vol, dvd)) / spot / (vol * np.sqrt(time))

def theta_call(strike, spot, rate, time, vol, dvd):
    return np.exp(-dvd * time) * -spot * stats.norm.pdf(d1(strike, spot, rate, time, vol, dvd)) * vol / 2 / np.sqrt(time)\
           - rate * strike * np.exp(-rate * time) * stats.norm.cdf(d2(strike, spot, rate, time, vol, dvd))\
           + dvd * spot * np.exp(-dvd * time) * stats.norm.cdf(d1(strike, spot, rate, time, vol, dvd))

def theta_put(strike, spot, rate, time, vol, dvd):
    return np.exp(-dvd * time) * -spot * stats.norm.pdf(d1(strike, spot, rate, time, vol, dvd)) * vol / 2 / np.sqrt(time)\
           + rate * strike * np.exp(-rate * time) * stats.norm.cdf(-d2(strike, spot, rate, time, vol, dvd))\
           - dvd * spot * np.exp(-dvd * time) * stats.norm.cdf(-d1(strike, spot, rate, time, vol, dvd))

def rho_call(strike, spot, rate, time, vol, dvd):
    return strike * time * np.exp(-rate * time) * stats.norm.cdf(d2(strike, spot, rate, time, vol, dvd))

def rho_put(strike, spot, rate, time, vol, dvd):
    return -strike * time * np.exp(-rate * time) * stats.norm.cdf(-d2(strike, spot, rate, time, vol, dvd))
