
import unittest
import numpy as np
import pandas as pd
import scipy
import toolkit as ftk

p = pd.Series([1, 1.25, 1, 0.8, 0.88], index=pd.date_range("2018-01-01", periods=5, freq="W"))
r = pd.Series([0.25, -0.2, -0.2, 0.1], index=pd.period_range("2018-01-08", periods=4, freq="W"))
n = 4

class TestFunctional(unittest.TestCase):

    def test_price_return_conversion(self):
        pd.testing.assert_series_equal(ftk.price_to_return(p), r)
        pd.testing.assert_series_equal(ftk.return_to_price(r), p)

    def test_compound_return(self):
        self.assertAlmostEqual(ftk.compound_return(p), -0.12)
        self.assertAlmostEqual(ftk.compound_return(r), -0.12)
        self.assertAlmostEqual(ftk.compound_return(r), (1+r).prod() - 1)
    
    def test_moment(self):
        self.assertAlmostEqual(ftk.skew(r), scipy.stats.skew(r, bias=False))
        self.assertAlmostEqual(ftk.skew(r), scipy.stats.skew(r) * np.sqrt((n - 1) * n) / (n - 2))
        self.assertAlmostEqual(ftk.kurt(r), scipy.stats.kurtosis(r, bias=False))
        self.assertAlmostEqual(ftk.kurt(r), (n - 1) / (n - 2) / (n - 3) * ((n + 1) * scipy.stats.kurtosis(r) + 6))
