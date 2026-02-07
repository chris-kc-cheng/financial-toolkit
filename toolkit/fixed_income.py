"""Fixed-income module

Vectorized. Interest rate can be a Series of floating rates.
"""
from typing import Tuple
import numpy as np
import pandas as pd
from scipy import optimize


def yield_to_maturity(
    price: float,
    coupon: float = 0.0,
    ttm: float = 1.0,
    freq: int = 2,
) -> float:
    """Yield to maturity

    Parameters
    ----------
    price : float
        Market price as a percentage of face value
    coupon : float, optional
        Annual coupon rate, by default 0.0 (i.e. zero coupon)
    ttm : float, optional
        Time to maturity in year, by default 1.0 (i.e. one year)
    freq : int, optional
        Coupon frequency, by default 2 (i.e. semi-annual coupon)

    Returns
    -------
    float
        Yield to maturity
    """
    return optimize.newton(lambda x: bond_price(x, coupon, ttm, freq) - price, 0)


def _toperiod(
    coupon: float = 0.0, ttm: float = 1.0, freq: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert coupon and time from annual to period

    Parameters
    ----------
    coupon : float, optional
        Annual coupon rate, by default 0.0 (i.e. zero coupon)
    ttm : float, optional
        Time to maturity in year, by default 1.0 (i.e. one year)
    freq : int, optional
        Coupon frequency, by default 2 (i.e. semi-annual coupon)

    Returns
    -------
    Tuple
        Tuple of a Series of period and a Series of cash flows
    """
    t = np.flip(np.arange(ttm * freq, 0, -1))
    cf = np.ones_like(t) * coupon / freq
    cf[-1] += 1
    return t, cf


def bond_price(
    y: float | np.ndarray, coupon: float = 0.0, ttm: float = 1.0, freq: int = 2
) -> float:
    """Present value of a bond as a percentage of face value

    Parameters
    ----------
    y : float | np.ndarray
        Yield to maturity, annualized
    coupon : float, optional
        Annual coupon rate, by default 0.0 (i.e. zero coupon)
    ttm : float, optional
        Time to maturity in year, by default 1.0 (i.e. one year)
    freq : int, optional
        Coupon frequency, by default 2 (i.e. semi-annual coupon)

    Returns
    -------
    float
        Present value of a bond as a percentage of face value
    """
    t, cf = _toperiod(coupon, ttm, freq)
    return (cf / (1 + y / freq) ** t).sum()


def duration_macaulay(
    y: float | np.ndarray, coupon: float = 0.0, ttm: float = 1.0, freq: int = 2
) -> float:
    """Macaulay duration, which is the average time to receive the cash flows
    weighted by the present value of cash flows.

    It assumes that coupon payments are reinvested continuously which is not
    the case.

    Parameters
    ----------
    y : float | np.ndarray
        Yield to maturity, annualized
    coupon : float, optional
        Annual coupon rate, by default 0.0 (i.e. zero coupon)
    ttm : float, optional
        Time to maturity in year, by default 1.0 (i.e. one year)
    freq : int, optional
        Coupon frequency, by default 2 (i.e. semi-annual coupon)

    Returns
    -------
    float
        Macaulay duration
    """
    t, cf = _toperiod(coupon, ttm, freq)
    dcf = cf / (1 + y / freq) ** t
    return (dcf * t / freq).sum() / dcf.sum()


def duration_modified(
    y: float | np.ndarray, coupon: float = 0.0, ttm: float = 1.0, freq: int = 2
) -> float:
    """Modified duration, adjusted for when actual payment frequency.

    Parameters
    ----------
    y : float | np.ndarray
        Yield to maturity, annualized
    coupon : float, optional
        Annual coupon rate, by default 0.0 (i.e. zero coupon)
    ttm : float, optional
        Time to maturity in year, by default 1.0 (i.e. one year)
    freq : int, optional
        Coupon frequency, by default 2 (i.e. semi-annual coupon)

    Returns
    -------
    float
        Modified duration
    """
    ytm = yield_to_maturity(bond_price(
        y, coupon, ttm, freq), coupon, ttm, freq)
    return duration_macaulay(y, coupon, ttm, freq) / (1 + ytm / freq)


def duration_effective(
    y: float | np.ndarray,
    coupon: float = 0.0,
    ttm: float = 1.0,
    freq: int = 2,
    delta=0.0025,
) -> float:
    """Effective duration

    Parameters
    ----------
    y : float | np.ndarray
        Yield to maturity, annualized
    coupon : float, optional
        Annual coupon rate, by default 0.0 (i.e. zero coupon)
    ttm : float, optional
        Time to maturity in year, by default 1.0 (i.e. one year)
    freq : int, optional
        Coupon frequency, by default 2 (i.e. semi-annual coupon)
    delta : float, optional
        Parallel shift in interest rate, by default 0.0025 (i.e. 25 bps)

    Returns
    -------
    float
        Effective duration
    """
    return (
        (
            bond_price(y - delta, coupon, ttm, freq)
            - bond_price(y + delta, coupon, ttm, freq)
        )
        / 2
        / bond_price(y, coupon, ttm, freq)
        / delta
    )


def convexity(
    y: float | np.ndarray, coupon: float = 0.0, ttm: float = 1.0, freq: int = 2
) -> float:
    """Convexity

    Parameters
    ----------
    y : float | np.ndarray
        Yield to maturity, annualized
    coupon : float, optional
        Annual coupon rate, by default 0.0 (i.e. zero coupon)
    ttm : float, optional
        Time to maturity in year, by default 1.0 (i.e. one year)
    freq : int, optional
        Coupon frequency, by default 2 (i.e. semi-annual coupon)

    Returns
    -------
    float
        Convexity
    """
    t, cf = _toperiod(coupon, ttm, freq)
    dcf = cf / (1 + y / freq) ** t
    return (dcf * t * (t + 1) / freq**2).sum() / dcf.sum()


def convexity_modified(
    y: float | np.ndarray, coupon: float = 0.0, ttm: float = 1.0, freq: int = 2
) -> float:
    """Modified convexity, adjusted for when actual payment frequency.

    Parameters
    ----------
    y : float | np.ndarray
        Yield to maturity, annualized
    coupon : float, optional
        Annual coupon rate, by default 0.0 (i.e. zero coupon)
    ttm : float, optional
        Time to maturity in year, by default 1.0 (i.e. one year)
    freq : int, optional
        Coupon frequency, by default 2 (i.e. semi-annual coupon)

    Returns
    -------
    float
        Modified convexity
    """
    ytm = yield_to_maturity(bond_price(
        y, coupon, ttm, freq), coupon, ttm, freq)
    return convexity(y, coupon, ttm, freq) / (1 + ytm / freq) ** 2


def convexity_effective(
    y: float | np.ndarray,
    coupon: float = 0.0,
    ttm: float = 1.0,
    freq: int = 2,
    delta=0.0025,
) -> float:
    """Effective convexity

    Parameters
    ----------
    y : float | np.ndarray
        Yield to maturity, annualized
    coupon : float, optional
        Annual coupon rate, by default 0.0 (i.e. zero coupon)
    ttm : float, optional
        Time to maturity in year, by default 1.0 (i.e. one year)
    freq : int, optional
        Coupon frequency, by default 2 (i.e. semi-annual coupon)
    delta : float, optional
        Parallel shift in interest rate, by default 0.0025 (i.e. 25 bps)

    Returns
    -------
    float
        Effective convexity
    """
    return (
        (
            bond_price(y - delta, coupon, ttm, freq)
            + bond_price(y + delta, coupon, ttm, freq)
            - 2 * bond_price(y, coupon, ttm, freq)
        )
        / bond_price(y, coupon, ttm, freq)
        / delta**2
    )


def cir(years: float, a: float, b: float, sigma: float, init: float, scenarios: int = 1, steps_per_year: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cox-Ingersoll-Ross model for interest rate simulation

    Parameters
    ----------
    years : float
        Number of years
    a : float
        Speed of mean reversion
    b : float
        Long-term average rate
    sigma : float
        Annualized volatility
    init : float
        Initial rate
    scenarios : int, optional
        Number of simulations, by default 1
    steps_per_year : int, optional
        Steps per year, by default 12

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Rates, and zero-coupon bond prices
    """

    init = np.log1p(init)
    dt = 1/steps_per_year
    n = int(years * steps_per_year) + 1

    shock = np.random.normal(0, scale=np.sqrt(dt), size=(n, scenarios))
    rates = np.empty_like(shock)
    rates[0] = init

    h = math.sqrt(a**2 + 2*sigma**2)
    prices = np.empty_like(shock)

    def price(ttm, r):
        _A = ((2*h*math.exp((h+a)*ttm/2)) /
              (2*h+(h+a)*(math.exp(h*ttm)-1)))**(2*a*b/sigma**2)
        _B = (2*(math.exp(h*ttm)-1))/(2*h + (h+a)*(math.exp(h*ttm)-1))
        _P = _A*np.exp(-_B*r)
        return _P

    prices[0] = price(years, init)

    for step in range(1, n):
        r_t = rates[step-1]
        d_r_t = a*(b-r_t)*dt + sigma*np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + d_r_t)
        prices[step] = price(years-step*dt, rates[step])

    rates = pd.DataFrame(data=np.expm1(rates), index=np.arange(n) * dt)
    prices = pd.DataFrame(data=prices, index=np.arange(n) * dt)
    return rates, prices
