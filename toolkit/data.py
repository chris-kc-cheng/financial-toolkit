"""Pull data from various sources

    - Yahoo! Finance
    - Ken French’s Data Library
    - Federal Reserve Economic Data
    - Bank of Canada
    - Statistics Canada
    - MSCI
    - Eurekahedge
"""
from functools import wraps
import io
import re
import datetime
import urllib
import requests
import pandas as pd
import pandas_datareader as pdr
import yfinance as yf


def _default_date(func):

    @wraps(func)
    def wrapper(**kwargs):
        if 'date' not in kwargs:
            kwargs['date'] = datetime.datetime.today().date()
        return func(**kwargs)

    wrapper.__doc__ = func.__doc__
    return wrapper


@_default_date
def last_business_day(date: datetime.date = None):
    return (date - datetime.timedelta(days=max(0, date.weekday() - 4)))


@_default_date
def end_of_month(date: datetime.date = None, n: int = 0) -> datetime.date:
    """Last day of the month that is `n` months after the specified date.
    Similar to the `EOMONTH` function in Excel.

    Parameters
    ----------
    date : datetime.date, optional
        A date, by default None (i.e. today)
    n : int, optional
        Number of months, by default 0. A positive value yields a month in the future,
        a negative value yields a month in the past.

    Returns
    -------
    datetime.date
        _description_
    """
    year = date.year + (date.month + n) // 12
    month = (date.month + n) % 12 + 1
    return datetime.date(year, month, 1) - datetime.timedelta(days=1)


def end_of_quarter(date: datetime.date = None, n: int = 0) -> datetime.date:
    """Last day of the calendar quarter that is `n` quarters after the specified date.

    Parameters
    ----------
    date : datetime.date, optional
        A date, by default None (i.e. today)
    n : int, optional
        Number of quarters, by default 0. A positive value yields a quarter in the future,
        a negative value yields a quarter in the past.

    Returns
    -------
    datetime.date
        _description_
    """
    return end_of_month(date, 2 - ((date.month - 1) % 3) + 3 * n)


def end_of_year(date: datetime.date = None, n: int = 0) -> datetime.date:
    """Last date of the calendar year that is `n` years after the specified date.

    Parameters
    ----------
    date : datetime.date, optional
        A date, by default None (i.e. today)
    n : int, optional
        Number of years, by default 0. A positive value yields a year in the future,
        a negative value yields a year in the past.

    Returns
    -------
    datetime.date
        _description_
    """
    return end_of_month(date, -date.month + 12 * n + 12)


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
def get_yahoo(ticker: str = "^GSPC", period: str = "max") -> pd.Series:
    """Download the historical adjusted closing price of a security with
    ticker `ticker`.

    Args:
        ticker (str): Yahoo! ticker of the security, by default "^GSPC", i.e. S&P 500
    period : str, optional
        Length of track record to download, by default 'max'

    Returns:
        pd.Series: Time series of the prices of the security
    """
    t = yf.Ticker(ticker)
    # Some securities e.g. crypto trades on weekends
    s = t.history(period=period)["Close"]
    s.name = ticker
    return s.asfreq("D")


def get_yahoo_bulk(tickers: list = ["^GSPC"], period: str = "max") -> pd.DataFrame:
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
    # Columns are already the tickers, note some securities like crypto trades in non-business days
    return yf.download(" ".join(tickers), auto_adjust=False, period=period)["Adj Close"].asfreq('D')


def get_msci(
        codes: list = [990100],
        end_date: str = None,
        fx: str = "USD",
        variant: str = "STRD",
        freq: str = "END_OF_MONTH",
        ror: bool = True) -> pd.DataFrame:
    """Download the historical index value of multiple indexes.

    Parameters
    ----------
    codes : list
        List of MSCI index code, by default [990100], i.e. MSCI World
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
    if end_date is None:
        end_date = last_business_day().strftime("%Y%m%d")
    url = f'https://app2-nv.msci.com/products/service/index/indexmaster/downloadLevelData?output=INDEX_LEVELS&currency_symbol={fx}&index_variant={variant}&start_date=19691231&end_date={end_date}&data_frequency={freq}&baseValue=false&index_codes={",".join(map(str, codes))}'
    df = pd.read_excel(url, thousands=',', parse_dates=[
                       0], skiprows=6, skipfooter=19).set_index('Date')
    if ror:
        df = df.pct_change().to_period('M')
    return df


def get_withintelligence(ids: list = []) -> pd.DataFrame:
    """Download the historical index value of selected hedge fund indexes.

    Returns
    -------
    pd.DataFrame
        Time series of the monthly returns. Index is PeriodIndex.
    """
    return None


def get_statcan_bulk(ids: list = [2062815], n: int = 25) -> pd.DataFrame:
    """Download the historical data from Statisticas Canada

    Parameters
    ----------
    ids : list
        List of vectors, by default [2062815]. (Vector number is a 10-digit number without leading "V")
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


def get_fred_bulk(ids: list = ["SOFR"], start_date: datetime.date = None, end_date: datetime.date = None) -> pd.DataFrame:
    """Download the historical Federal Reserve Economic Data (FRED) from Frederal Reserver Bank of St. Louis

    Parameters
    ----------
    ids : list, optional
        List of series Ids, by default ["SOFR"], i.e. Secured Overnight Financing Rate
    start : datetime.date, optional
        Start date, by default 2 years before today
    end : datetime.date, optional
        End date, by default today

    Returns
    -------
    pd.DataFrame
        Index is DatetimeIndex (freq='ME')
    """
    if start_date is None:
        start_date = end_of_month(n=-25)
    if end_date is None:
        end_date = datetime.datetime.today()
    df = pdr.DataReader(ids, 'fred', start_date, end_date)
    return df.groupby(pd.Grouper(freq="ME")).last()


def get_us_yield_curve(year: int = 0, n: int = 2) -> pd.DataFrame:
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
    if year == 0:
        year = datetime.datetime.today().year
    data = [requests.get(
        f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{y}/all?type=daily_treasury_yield_curve&_format=csv", verify=False).text for y in range(year, year - n, -1)]
    df = pd.concat([pd.read_csv(io.StringIO(d), index_col=0)
                   for d in data], axis=0)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def get_boc_bulk(ids: list = ["V80691342"], start_date: datetime.date = None, end_date: datetime.date = None) -> pd.DataFrame:
    """Download the historical data from Bank of Canada

    Parameters
    ----------
    ids : list
        List of series names, by default ["V80691342"], i.e. 1 month Treasury bill
    start : datetime.date, optional
        Start date, by default end_of_month(n=-25)
    end : datetime.date, optional
        End date, by default end_of_month()

    Returns
    -------
    pd.DataFrame
        _description_
    """
    if start_date is None:
        start_date = end_of_month(n=-25)
    if end_date is None:
        end_date = end_of_month()
    url = f'https://www.bankofcanada.ca/valet/observations/{urllib.parse.quote(",".join(ids))}/csv?start_date={start_date.strftime("%Y-%m-%d")}&end_date={end_date.strftime("%Y-%m-%d")}'
    csv = requests.get(url, verify=False)
    return pd.read_csv(io.StringIO(csv.text.split('"OBSERVATIONS"\r\n')[1]), parse_dates=["date"]).set_index("date").loc[:, ids]


def get_spglobal_bulk(codes: list = [5457755]) -> pd.DataFrame:
    """Download the historical index values from S&P Global

    Parameters
    ----------
    codes : list, optional
        List of index code, by default [5457755]

    Returns
    -------
    pd.DataFrame
        Index is DatetimeIndex (freq='B')
    """
    url = f"https://www.spglobal.com/spdji/en/util/redesign/get-index-comparison-data.dot?compareArray={'&compareArray='.join([str(i) for i in codes])}&periodFlag=tenYearFlag&language_id=1"
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
        "x-requested-with": "XMLHttpRequest"
    }
    json = requests.get(url, headers=headers).json()
    names = {str(i["indexId"]): i["indexName"]
             for i in json["performanceComparisonHolder"]["indexPerformanceForComparison"]}
    values = json["levelComparisonHolder"]["indexLevelForComparison"]
    df = pd.DataFrame([j for i in list(names.keys()) for j in values[i]])
    df["date"] = df["effectiveDate"].apply(
        lambda e: datetime.datetime.fromtimestamp(e / 1000))
    df = df.pivot(index="date", columns="indexId",
                  values="indexValue").asfreq("B")
    df.columns = pd.MultiIndex.from_tuples(
        [(i, names[str(i)]) for i in list(df.columns)])
    return df
