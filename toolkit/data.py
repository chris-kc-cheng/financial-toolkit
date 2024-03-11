"""Pull data from various sources

    - Yahoo! Finance
    - Ken French’s Data Library
    - MSCI
    - Eurekahedge
"""
import re
import datetime
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
    pattern = re.compile(r"^\w+_Factors\w*$")
    full = pdr.famafrench.get_available_datasets()
    return sorted([x for x in full if pattern.match(x)])


def get_famafrench_factors(dataset: str, add_momentum: bool = False) -> pd.DataFrame:
    freq = "B" if "aily" in dataset else "M"
    data = pdr.get_data_famafrench(
        dataset, start="1990-01-01")[0].asfreq(freq) / 100
    factors = data.iloc[:, :-1]
    rfr = data.iloc[:, -1]
    if add_momentum:
        # Momentum has less data
        factors = factors.join(
            pdr.get_data_famafrench(
                re.sub(r"[35]_Factors", "Mom_Factor", dataset), start="1990-01-01"
            )[0].asfreq(freq)
            / 100,
            how="inner",
        )
    return factors.join(rfr, how="inner")


# Yahoo
def get_yahoo(ticker: str) -> pd.Series:
    """Download the historical adjusted closing price of a security with
    ticker `ticker`.

    Args:
        ticker (str): Yahoo! ticker of the security

    Returns:
        pd.Series: Time series of the prices of the security
    """
    t = yf.Ticker(ticker)
    s = t.history(period="max")["Close"].asfreq("B")
    s.name = ticker
    return s


def get_yahoo_bulk(tickers: list, period: str = "max") -> pd.DataFrame:
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
    # Columns are already the tickers
    return yf.download(" ".join(tickers), period=period)["Adj Close"].asfreq("B")


def get_last_business_day(d=datetime.datetime.today()):
    return (d - datetime.timedelta(days=max(0, d.weekday() - 4))).date()


def get_msci(
        codes: list,
        end_date: str = get_last_business_day().strftime("%Y%m%d"),
        fx: str = "USD",
        variant: str = "STRD",
        freq: str = "END_OF_MONTH",
        ror: bool = True) -> pd.DataFrame:
    """Download the historical index value of multiple indexes.

    Parameters
    ----------
    codes : list
        List of MSCI index code.
        See https://www.msci.com/our-solutions/indexes/index-resources/index-tools
    end_date : str, optional
        As of date, by default get_last_business_day().strftime("%Y%m%d")
    fx : str, optional
        Currency, by default "USD"
    variant : str, optional
        Index Level ('STRD' for Price, 'NETR' for Net, 'GRTR' for Gross), by default "STRD"
    freq : str, optional
        Data Frequency, by default "END_OF_MONTH"
    ror : bool, optional
        Convert from index value to return, by default True

    Returns
    -------
    pd.DataFrame
        Time series of the index values. Index is DatetimeIndex (freq=None).
    """
    url = f'https://app2-nv.msci.com/products/service/index/indexmaster/downloadLevelData?output=INDEX_LEVELS&currency_symbol={fx}&index_variant={variant}&start_date=19691231&end_date={end_date}&data_frequency={freq}&baseValue=false&index_codes={",".join(map(str, codes))}'
    df = pd.read_excel(url, thousands=',', parse_dates=[
                       0], skiprows=6, skipfooter=19).set_index('Date')
    if ror:
        df = df.pct_change().to_period('M')
    return df


def get_eurekahedge() -> pd.DataFrame:
    """Download the historical index value of all hedge fund indexes.

    Returns
    -------
    pd.DataFrame
        Time series of the monthly returns. Index is PeriodIndex.
    """
    return pd.read_csv("https://www.eurekahedge.com/df/Eurekahedge_indices.zip", parse_dates=['Date'], index_col=[0, 1, 2], na_values=' ').squeeze().unstack().T.to_period('M') / 100.0
