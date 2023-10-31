import re
import pandas as pd
import pandas_datareader as pdr
import yfinance as yf

# Eurekahedge

# Fama French
def get_famafrench_datasets():    
    pattern =  re.compile(r'^\w+_Factors\w*$')
    full = pdr.famafrench.get_available_datasets()
    return sorted([x for x in full if pattern.match(x)])

def get_famafrench_factors(dataset, add_momentum: bool = False):
    freq = 'B' if 'aily' in dataset else 'M'
    print('HERE', freq)
    factors = pdr.get_data_famafrench(dataset, start='1990-01-01')[0].asfreq(freq) / 100
    if add_momentum:
        # Momentum has less data
        factors = factors.join(pdr.get_data_famafrench(re.sub(r'[35]_Factors', 'Mom_Factor', dataset), start='1990-01-01')[0].asfreq(freq) / 100, how='inner')
    return factors

# Yahoo
def get_yahoo(ticker):
    t = yf.Ticker(ticker)
    return t.history(period='max')['Close'].asfreq('B')
