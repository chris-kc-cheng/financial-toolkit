import unittest
import numpy as np
import toolkit as ftk

# Alternative implementations using explicit formula
# However, those are not as flexible as the ndarray implementation due to
# the restriction in time to maturity

def bond_price_alt(
    y, coupon: float = 0.0, ttm: float = 1.0, freq: int = 2
) -> float:
    """Explicit formula when all coupon payments are identical (which is normal)
    """
    return coupon / freq * (1 - 1 / (1 + y / freq) ** (ttm * freq)) / (
        y / freq
    ) + 1 / (1 + y / freq) ** (ttm * freq)

def duration_macaulay_alt(
    yper, coupon: float = 0.0, ttm: float = 1.0, freq: int = 2
):
    """Explicit formula when all coupon payments are identical (which is normal)
    """
    n = ttm * freq
    y = yper / freq
    return ((1 + y) / freq / y - (1 + y + n * (coupon / freq - y)) / (coupon * ((1 + y) ** n - 1) + freq * y))

class TestBonds(unittest.TestCase):

    def test_yield_to_maturity(self):
        # Par bond
        self.assertAlmostEqual(ftk.bond_price(0.09, 0.09, 5), 1)

        # Luenberger
        self.assertAlmostEqual(ftk.bond_price(0.05, 0.09, 1), 1.0385, 4)
        self.assertAlmostEqual(ftk.bond_price(0.15, 0.09, 30), 0.6052, 4)

        # Investopedia
        self.assertAlmostEqual(ftk.bond_price(0.068, 0.05, 2.5, 2), 0.9592, 4)
        self.assertAlmostEqual(ftk.yield_to_maturity(0.9592, 0.05, 2.5, 2), 0.068, 3)

        # Tuckman
        self.assertAlmostEqual(
            ftk.bond_price(0.000252, 0.07625, 1.5, 2), 1.11396, 4
        )
        self.assertAlmostEqual(
            ftk.yield_to_maturity(1.113969, 0.07625, 1.5, 2), 0.000252, 6
        )

        # Alternative implementation
        self.assertAlmostEqual(
            ftk.bond_price(0.000252, 0.07625, 1.5, 2),
            bond_price_alt(0.000252, 0.07625, 1.5, 2)
        )

    def test_duration(self):

        # Luenberger
        self.assertAlmostEqual(ftk.duration_macaulay(0.1, 0.1, 30, 2), 9.938, 3)
        self.assertAlmostEqual(ftk.duration_macaulay(0.05, 0.01, 1, 2), 0.997, 3)
        self.assertAlmostEqual(ftk.duration_macaulay(0.05, 0.1, 100, 2), 20.067, 3)
        
        # Investopedia
        self.assertAlmostEqual(ftk.duration_macaulay(0.06, 0.1, 3, 2), 2.6840, 4)

        # Bacon
        ir = np.concatenate([
            np.repeat(0.035, 8),
            np.repeat(0.0375, 8),
            np.repeat(0.04, 4)])

        self.assertAlmostEqual(ftk.duration_macaulay(0.0389, 0.06, 20, 1), 13.13, 2)
        self.assertAlmostEqual(ftk.duration_modified(0.0389, 0.06, 20, 1), 12.64, 2)
        self.assertAlmostEqual(ftk.duration_effective(ir, 0.06, 20, 1, 0.0025), 12.55, 2)
        self.assertAlmostEqual(ftk.convexity(ir, 0.06, 20, 1), 230.06, 2)
        self.assertAlmostEqual(ftk.convexity_modified(ir, 0.06, 20, 1), 213.14, 2)
        self.assertAlmostEqual(ftk.convexity_effective(ir, 0.06, 20, 1, 0.0025), 213.00, 2)

        # Alternative implementation
        self.assertAlmostEqual(
            ftk.duration_macaulay(0.1, 0.1, 30, 2),
            duration_macaulay_alt(0.1, 0.1, 30, 2)
        )