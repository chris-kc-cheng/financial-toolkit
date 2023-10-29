
import unittest
import numpy as np
import pandas as pd
import scipy
import toolkit as ftk

n = 4
p = pd.Series([1, 1.25, 1, 0.8, 0.88], index=pd.date_range("2018-01-01", periods=5, freq="W"))
r = pd.Series([0.25, -0.2, -0.2, 0.1], index=pd.period_range("2018-01-08", periods=4, freq="W"))
pp = pd.concat([p, p], axis=1)
rr = pd.concat([r, r], axis=1)

class TestFunctional(unittest.TestCase):

    unit_price = pd.Series([100,100.3,102.91,104.04,103.1,104.55,107.06,108.66,115.83,114.21,118.67,118.07,127.64,132.74,127.83,120.03,121.71,115.75,113.32,120.34,127.32,119.18,121.2,120.72,120.48,117.95,119.24,124.85,127.84,132.06,131.14,137.3,138.13,139.51,139.23,143.96,145.4
], index=pd.date_range('1999-12', periods=37, freq='M'))
    benchmark_price = pd.Series([100,100.2,102.71,104.55,103.4,104.85,107.26,108.76,115.83,114.1,118.89,118.53,128.37,133.38,128.31,120.35,122.16,116.29,113.97,120.81,127.57,119.03,121.29,120.92,120.8,117.66,118.48,123.58,127.16,132,131.73,138.45,140.39,142.21,142.64,147.49,150.59],
                                index=pd.date_range('1999-12', periods=37, freq='M'))
    price_df = pd.concat([unit_price, benchmark_price], axis=1)
    

    def test_periodicity(self):
        for p in ftk.PERIODICITY:
            s = pd.Series([0], index=pd.period_range("2000-01-01", periods=1, freq=p))
            self.assertEqual(ftk.periodicity(s), ftk.PERIODICITY[p])
            df = pd.concat([s, s], axis=1)
            self.assertEqual(ftk.periodicity(df), ftk.PERIODICITY[p])

    def test_price_return_conversion(self):
        pd.testing.assert_series_equal(ftk.price_to_return(p), r)
        pd.testing.assert_series_equal(ftk.return_to_price(r), p)
        pd.testing.assert_frame_equal(ftk.price_to_return(pp), rr)
        pd.testing.assert_frame_equal(ftk.return_to_price(rr), pp)

    def test_decorator(self):        
        
        @ftk.functional.requireReturn
        def rfunc(x):
            return x
        
        @ftk.functional.requirePrice
        def pfunc(x):
            return x
        
        pd.testing.assert_series_equal(rfunc(r), r)
        pd.testing.assert_series_equal(rfunc(p), r)
        pd.testing.assert_series_equal(pfunc(r), p)
        pd.testing.assert_series_equal(pfunc(p), p)

    def test_compound_return(self):
        self.assertAlmostEqual(ftk.compound_return(p, False), -0.12)
        self.assertAlmostEqual(ftk.compound_return(r, False), -0.12)
        self.assertAlmostEqual(ftk.compound_return(r, False), (1+r).prod() - 1)
        self.assertAlmostEqual(ftk.compound_return(pp, False)[0], -0.12)
        self.assertAlmostEqual(ftk.compound_return(rr, False)[1], -0.12)
    
    def test_skew_kurt(self):
        self.assertAlmostEqual(ftk.skew(r), scipy.stats.skew(r, bias=False))        
        self.assertAlmostEqual(ftk.skew(r), scipy.stats.skew(r) * np.sqrt((n - 1) * n) / (n - 2))
        np.testing.assert_array_almost_equal(ftk.skew(rr), scipy.stats.skew(rr, bias=False))

        self.assertAlmostEqual(ftk.kurt(r), scipy.stats.kurtosis(r, bias=False))        
        self.assertAlmostEqual(ftk.kurt(r), (n - 1) / (n - 2) / (n - 3) * ((n + 1) * scipy.stats.kurtosis(r) + 6))
        np.testing.assert_array_almost_equal(ftk.kurt(rr), scipy.stats.kurtosis(rr, bias=False))

    def test_mean(self):
        self.assertAlmostEqual(ftk.arithmetic_mean(r), r.sum() / len(r))
        np.testing.assert_array_almost_equal(ftk.arithmetic_mean(rr), rr.sum() / len(rr))
        
        self.assertAlmostEqual(ftk.geometric_mean(r), scipy.stats.gmean(1 + r) - 1)
        np.testing.assert_array_almost_equal(ftk.geometric_mean(rr), scipy.stats.gmean(1 + rr) - 1)
        
    def test_descriptive_stats(self):        
        self.assertAlmostEqual(ftk.arithmetic_mean(self.unit_price), 0.011, 3)
        self.assertAlmostEqual(ftk.compound_return(self.unit_price), 0.454, 3)
        self.assertAlmostEqual(ftk.compound_return(self.unit_price, annualize=True), 0.1329, 4)
        self.assertAlmostEqual(ftk.mean_abs_dev(self.unit_price), 0.0252, 4)
        self.assertAlmostEqual(ftk.variance(self.unit_price), 0.0011 * 36 / 35, 4)
        self.assertAlmostEqual(ftk.volatility(self.unit_price), 0.0336, 4)
        self.assertAlmostEqual(ftk.volatility(self.unit_price, annualize=True), 0.1149 * np.sqrt(36 / 35), 4)
        self.assertAlmostEqual(ftk.skew(self.unit_price), -0.25, 2)
        self.assertAlmostEqual(ftk.kurt(self.unit_price), 0.16, 2)
        self.assertAlmostEqual(ftk.covariance(self.price_df).iloc[0, 1], 0.001109 * 36 / 35, 6)
        self.assertAlmostEqual(ftk.correlation(self.price_df).iloc[0, 1], 0.995, 3)
        self.assertAlmostEqual(ftk.sharpe(self.unit_price, 0.0243), 0.93, 2)

    # Add benchmark regression test

    def test_drawdown(self):
        # Assuming uninterrupted drawdown definition is used
        self.assertAlmostEqual(ftk.worst_drawdown(self.unit_price), -0.1463, 4)
        self.assertAlmostEqual(ftk.calmar(self.unit_price, rfr=0.0243), 0.74, 2)
        self.assertAlmostEqual(ftk.avg_drawdown(self.unit_price, d=3), -0.056, 3)
        #ftk.sterling() # Original
        self.assertAlmostEqual(ftk.sterling_modified(self.unit_price, rfr=0.0243, d=3), 1.92, 2)
        self.assertAlmostEqual(ftk.drawdown_deviation(self.unit_price, d=3), -0.0245, 4)
        self.assertAlmostEqual(ftk.burke_modified(self.unit_price, rfr=0.0243, d=3), 4.43, 1) # vs 4.42
        self.assertAlmostEqual(ftk.avg_annual_drawdown(self.unit_price), -0.0604, 4)
        self.assertAlmostEqual(ftk.sterling_calmar(self.unit_price, rfr=0.0243, d=3), 1.80, 2)
        self.assertAlmostEqual(ftk.pain_index(self.unit_price), 0.0376, 4)
        self.assertAlmostEqual(ftk.pain(self.unit_price, 0.0243), 2.89, 2)
        self.assertAlmostEqual(ftk.ulcer_index(self.unit_price), 0.0597, 4)
        self.assertAlmostEqual(ftk.martin(self.unit_price, 0.0243), 1.82, 2)

    def test_partial_moments(self):
        self.assertAlmostEqual(ftk.downside_potential(self.unit_price, mar=0.005), 0.0101, 4)
        self.assertAlmostEqual(ftk.downside_risk(self.unit_price, mar=0.005), 0.0212, 4)
        self.assertAlmostEqual(ftk.downside_risk(self.unit_price, mar=0.005, annualize=True), 0.0733, 4)
        self.assertAlmostEqual(ftk.upside_potential(self.unit_price, mar=0.005), 0.0161, 4)
        self.assertAlmostEqual(ftk.upside_risk(self.unit_price, mar=0.005), 0.0262, 4)
        self.assertAlmostEqual(ftk.upside_risk(self.unit_price, mar=0.005, annualize=True), 0.0909, 4)
        self.assertAlmostEqual(ftk.omega(self.unit_price, mar=0.005), 1.6, 1)
        self.assertAlmostEqual(ftk.upside_potential_ratio(self.unit_price, 0.005), 0.22, 2)
        self.assertAlmostEqual(ftk.variability_skewness(self.unit_price, 0.005), 1.24, 2)
        self.assertAlmostEqual(ftk.sortino(self.unit_price, mar=0.005), 0.97, 2)
