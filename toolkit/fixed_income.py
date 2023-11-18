"""Fixed-income module

"""
import numpy as np
from scipy import optimize


def yield_to_maturity(
    price: float,
    face: float = 100.0,
    coupon: float = 0.0,
    ttm: float = 1.0,
    freq: int = 2,
) -> float:
    """Yield to maturity

    Parameters
    ----------
    price : float
        Market price of the bond
    face : float, optional
        Face value, by default 100.0
    coupon : float, optional
        Annual coupon in dollar terms, by default 0.0 (i.e. zero coupon)
    ttm : float, optional
        Time to maturity, by default 1.0 (i.e. one year)
    freq : int, optional
        Coupon frequency, by default 2 (i.e. semi-annual coupon)

    Returns
    -------
    float
        Yield to maturity
    """
    return optimize.newton(lambda x: bond_price(x, face, coupon, ttm, freq) - price, 0)


def bond_price(        
    y, face: float = 100.0, coupon: float = 0.0, ttm: float = 1.0, freq: int = 2
) -> float:
    """Present value of a bond.

    Parameters
    ----------
    price : float
        Market price of the bond
    face : float, optional
        Face value, by default 100.0
    coupon : float, optional
        Annual coupon in dollar terms, by default 0.0 (i.e. zero coupon)
    ttm : float, optional
        Time to maturity, by default 1.0 (i.e. one year)
    freq : int, optional
        Coupon frequency, by default 2 (i.e. semi-annual coupon)

    Returns
    -------
    float
        Present value of a bond
    """
    t = np.arange(ttm * freq, 0, -1)
    cf = np.ones_like(t) * coupon / freq
    cf[0] += face
    return (cf / (1 + y / freq) ** t).sum()
