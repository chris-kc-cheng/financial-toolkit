import unittest
import numpy as np
import toolkit as ftk

class TestBonds(unittest.TestCase):
    
    def test_yield_to_maturity(self):

        # Par bond
        self.assertAlmostEqual(ftk.bond_price(0.09, 100, 9, 5), 100)

        # Luenberger
        self.assertAlmostEqual(ftk.bond_price(0.05, 100, 9, 1), 103.85, 2)
        self.assertAlmostEqual(ftk.bond_price(0.15, 100, 9, 30), 60.52, 2)

        # Investopedia
        self.assertAlmostEqual(ftk.bond_price(0.068, 100.0, 5, 2.5, 2), 95.92, 2)
        self.assertAlmostEqual(ftk.yield_to_maturity(95.92, 100.0, 5, 2.5, 2), 0.068, 3)

        # Tuckman
        self.assertAlmostEqual(ftk.bond_price(0.000252, 100.0, 7.625, 1.5, 2), 111.396, 2)
        self.assertAlmostEqual(ftk.yield_to_maturity(111.3969, 100, 7.625, 1.5, 2), 0.000252, 6)