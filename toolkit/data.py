"""Pull data from various sources

    - Yahoo! Finance
    - Ken French’s Data Library
    - Eurekahedge
"""
import re
import pandas as pd
import pandas_datareader as pdr
import yfinance as yf

# Eurekahedge

# Fama French
def get_famafrench_datasets() -> list:
    """Get the list of datasets related to factor analysis

    Returns:
        list: List of dataset names in str
    """
    pattern =  re.compile(r'^\w+_Factors\w*$')
    full = pdr.famafrench.get_available_datasets()
    return sorted([x for x in full if pattern.match(x)])

def get_famafrench_factors(dataset: str, add_momentum: bool = False) -> pd.DataFrame:
    freq = 'B' if 'aily' in dataset else 'M'
    data = pdr.get_data_famafrench(dataset, start='1990-01-01')[0].asfreq(freq) / 100
    factors = data.iloc[:, :-1]
    rfr = data.iloc[:, -1]
    if add_momentum:
        # Momentum has less data
        factors = factors.join(pdr.get_data_famafrench(re.sub(r'[35]_Factors', 'Mom_Factor', dataset), start='1990-01-01')[0].asfreq(freq) / 100, how='inner')
    return factors.join(rfr, how='inner')

# Yahoo
def get_yahoo(ticker: str) -> pd.Series:
    """Download the historical adjusted closing price of a security with
    ticker `ticker`

    Args:
        ticker (str): Yahoo! ticker of the security

    Returns:
        pd.Series: Time series of the prices of the security
    """
    t = yf.Ticker(ticker)
    s = t.history(period='max')['Close'].asfreq('B')
    s.name = ticker
    return s

def get_yahoo_bulk(tickers: list, period: str = 'max') -> pd.DataFrame:
    """Download the historical adjusted closing price of multiple securities
    with ticker in the `tickers` list.

    Parameters
    ----------
    tickers : list
        List of Yahoo! tickers
    period : str, optional
        Length of track record to download, by default 'max'

    Returns
    -------
    pd.DataFrame
        Time series of the prices of the securities
    """
    return yf.download(' '.join(tickers), period=period)['Adj Close'].asfreq('B')
