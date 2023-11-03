import unittest
import pandas as pd
import pandas_datareader as pdr
import toolkit as ftk

class TestFamaFrench(unittest.TestCase):
    
    def test_regression(self):        
        factors = ftk.get_famafrench_factors('F-F_Research_Data_Factors')

        alpha = 0.05
        loadings = pd.Series([0.4, 0.3, 0.2, 0.1], index=factors.columns)
        portfolio = (loadings * factors).sum(axis=1) + alpha

        slopes = ftk.beta(portfolio, factors)
        intercept = ftk.alpha(portfolio, factors)

        pd.testing.assert_series_equal(slopes, loadings)
        self.assertAlmostEqual(intercept, alpha)
    
    def test_momentum(self):        
        factors = ftk.get_famafrench_factors('Developed_5_Factors', add_momentum=True)

        alpha = 0.05
        loadings = pd.Series([0.25, 0.2, 0.15, 0.1, 0.05, 0.05, 0.2], index=factors.columns)
        portfolio = (loadings * factors).sum(axis=1) + alpha

        slopes = ftk.beta(portfolio, factors)
        intercept = ftk.alpha(portfolio, factors)

        pd.testing.assert_series_equal(slopes, loadings)
        self.assertAlmostEqual(intercept, alpha)
        