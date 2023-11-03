import unittest
import numpy as np
import pandas as pd
import toolkit as ftk

class TestPortfolio(unittest.TestCase):

    def setUp(self):

        self.n = 10  # periods
        self.m = 5   # asset classes
        self.k = 3   # weighting schemes
        self.name = list('abcde')

        self.rs = pd.DataFrame(np.random.rand(self.m * self.n).reshape(self.n, self.m), index=np.arange(self.n), columns=self.name) # DataFrame (n, m)
        self.r = ftk.compound_return(self.rs) # Series (m,)
        self.cov = ftk.covariance(self.rs) # DataFrame (m, m)
        self.w = pd.Series(np.repeat(1/self.m, self.m), index=self.name) # Series (m,)
        self.ws = pd.DataFrame({0: self.w, 1: self.w+0.1, 2: self.w+0.2}) # DataFrame (m, k)

        

    def test_dimension(self):

        self.assertIsInstance(ftk.portfolio_return(self.w, self.r), np.float64)
        self.assertEqual(ftk.portfolio_return(self.ws, self.r).shape, (self.k,))
        self.assertEqual(ftk.portfolio_return(self.w, self.rs).shape, (self.n,))
        self.assertEqual(ftk.portfolio_return(self.ws, self.rs).shape, (self.n, self.k))
        
        self.assertIsInstance(ftk.portfolio_volatility(self.w, self.cov), np.float64)
        self.assertEqual(ftk.portfolio_volatility(self.ws, self.cov).shape, (self.k,))
        
        self.assertEqual(ftk.risk_contribution(self.w, self.cov).shape, (self.m,))
        self.assertEqual(ftk.risk_contribution(self.ws, self.cov).shape, (self.k, self.m))

    def test_risk_parity(self):

        rc = ftk.risk_contribution(ftk.risk_parity(self.cov), self.cov)
        self.assertAlmostEqual(rc.min(), rc.max(), 3)
        