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
    return spot * np.exp(-dvd * time) * stats.norm.cdf(d1(strike, spot, rate, time, vol, dvd))\
           - strike * np.exp(-rate * time) * stats.norm.cdf(d2(strike, spot, rate, time, vol, dvd))

def price_put(strike, spot, rate, time, vol, dvd):
    return strike * np.exp(-rate * time) * stats.norm.cdf(-d2(strike, spot, rate, time, vol, dvd))\
        - spot * np.exp(-dvd * time) * stats.norm.cdf(-d1(strike, spot, rate, time, vol, dvd))

def delta_call(strike, spot, rate, time, vol, dvd):
    return np.exp(-dvd * time) * stats.norm.cdf(d1(strike, spot, rate, time, vol, dvd))

def delta_put(strike, spot, rate, time, vol, dvd):
    return delta_call(strike, spot, rate, time, vol, dvd) - 1

# Scaled to 1%
def vega(strike, spot, rate, time, vol, dvd):
    return spot * np.exp(-dvd * time) * stats.norm.pdf(d1(strike, spot, rate, time, vol, dvd)) * np.sqrt(time) / 100

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