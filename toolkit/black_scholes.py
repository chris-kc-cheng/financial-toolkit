import numpy as np
from scipy import stats

def bp(face, rate, time):
        return face * np.exp(-rate * time)

def d1(strike, spot, rate, time, vol, dvd):
    return (np.log(spot / strike) + (rate - dvd + vol ** 2 / 2) * time) / (vol * np.sqrt(time))

def d2(strike, spot, rate, time, vol, dvd):
    return d1(strike, spot, rate, time, vol, dvd) - vol * np.sqrt(time)

def value(iscall, strike, spot, rate, time, vol, dvd):
    sign = 1 if iscall else -1
    return sign * spot * np.exp(-dvd * time) * stats.norm.cdf(sign * d1(strike, spot, rate, time, vol, dvd))\
           - sign * strike * np.exp(-rate * time) * stats.norm.cdf(sign * d2(strike, spot, rate, time, vol, dvd))

def delta(iscall, strike, spot, rate, time, vol, dvd):
    sign = 1 if iscall else -1
    return sign * np.exp(-dvd * time) * stats.norm.cdf(sign * d1(strike, spot, rate, time, vol, dvd))

# Scaled to 1%
def vega(strike, spot, rate, time, vol, dvd):
    return spot * np.exp(-dvd * time) * stats.norm.pdf(d1(strike, spot, rate, time, vol, dvd)) * np.sqrt(time) / 100

def gamma(strike, spot, rate, time, vol, dvd):
    return np.exp(-dvd * time) * stats.norm.pdf(d1(strike, spot, rate, time, vol, dvd)) / spot / (vol * np.sqrt(time))

def theta(iscall, strike, spot, rate, time, vol, dvd):
    sign = 1 if iscall else -1
    return np.exp(-dvd * time) * -spot * stats.norm.pdf(d1(strike, spot, rate, time, vol, dvd)) * vol / 2 / np.sqrt(time)\
           - sign * rate * strike * np.exp(-rate * time) * stats.norm.cdf(sign * d2(strike, spot, rate, time, vol, dvd))\
           + sign * dvd * spot * np.exp(-dvd * time) * stats.norm.cdf(sign * d1(strike, spot, rate, time, vol, dvd))

def rho(iscall, strike, spot, rate, time, vol, dvd):
    sign = 1 if iscall else -1
    return sign * strike * time * np.exp(-rate * time) * stats.norm.cdf(sign * d2(strike, spot, rate, time, vol, dvd))