"""Pull data from various sources

    - Yahoo! Finance
    - Ken French’s Data Library
    - MSCI
    - Eurekahedge
"""
import re
import datetime
import requests
import pandas as pd
import pandas_datareader as pdr
import yfinance as yf

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
    # Some securities e.g. crypto trades on weekends
    s = t.history(period="max")["Close"]
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


def get_statcan_bulk(ids: list, n: int = 24) -> pd.DataFrame:
    """Download the historical data from Statisticas Canada

    Parameters
    ----------
    ids : list
        List of vectors (10 digits without leading "V")
    n : int, optional
        Number of months, by default 24

    Returns
    -------
    pd.DataFrame
        Column names are ids. Index is DatetimeIndex (freq=None).
    """
    url = "https://www150.statcan.gc.ca/t1/wds/rest/getDataFromVectorsAndLatestNPeriods"
    param = [{"vectorId": id, "latestN": n} for id in ids]
    json = requests.post(url, json=param).json()
    data = [pd.DataFrame(json[i]["object"]["vectorDataPoint"]).set_index(
        "refPer")["value"] for i in range(len(json))]
    df = pd.concat(data, axis=1, keys=ids)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def get_fred_bulk(ids: list = [], start=datetime.datetime.today() - datetime.timedelta(days=2 * 365), end=datetime.datetime.today()) -> pd.DataFrame:
    """Download the historical Federal Reserve Economic Data (FRED) from Frederal Reserver Bank of St. Louis

    Parameters
    ----------
    ids : list, optional
        List of series Ids, by default []
    start : _type_, optional
        Start date, by default 2 years before today
    end : _type_, optional
        End date, by default today

    Returns
    -------
    pd.DataFrame
        Index is DatetimeIndex (freq='ME')
    """
    df = pdr.DataReader(['SOFR', 'T10YIE'], 'fred', start, end)
    return df.groupby(pd.Grouper(freq="ME")).last()


def get_us_yield_curve(year: int = datetime.datetime.today().year, n: int = 2) -> pd.DataFrame:
    """Download the historical Treasury Par Yield Curve Rates

    Parameters
    ----------
    year : int, optional
        Year, by default the current year
    n : int, optional
        Number of years, by default 2

    Returns
    -------
    pd.DataFrame
        Index is DatetimeIndex (freq=None)
    """
    data = [pd.read_csv(
        f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{y}/all?type=daily_treasury_yield_curve&_format=csv", index_col=0) for y in range(year, year - n, -1)]
    df = pd.concat(data, axis=0)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()
