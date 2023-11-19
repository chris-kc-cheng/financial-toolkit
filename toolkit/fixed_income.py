"""Fixed-income module

Vectorized. Interest rate can be a Series of floating rates.
"""
from typing import Tuple
import numpy as np
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
    ytm = yield_to_maturity(bond_price(y, coupon, ttm, freq), coupon, ttm, freq)
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
    ytm = yield_to_maturity(bond_price(y, coupon, ttm, freq), coupon, ttm, freq)
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
