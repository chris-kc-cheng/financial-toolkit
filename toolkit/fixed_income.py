"""Fixed-income module

"""
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
):
    t = np.arange(ttm * freq, 0, -1)
    cf = np.ones_like(t) * coupon / freq
    cf[0] += 1
    return t, cf


def bond_price(
    y, coupon: float = 0.0, ttm: float = 1.0, freq: int = 2
) -> float:
    """Present value of a bond as a percentage of face value

    Parameters
    ----------
    y : float
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
    y, coupon: float = 0.0, ttm: float = 1.0, freq: int = 2
) -> float:
    t, cf = _toperiod(coupon, ttm, freq)
    dcf = cf / (1 + y / freq) ** t
    return (dcf * t / freq).sum() / dcf.sum()
