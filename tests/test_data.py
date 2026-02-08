import unittest
import numpy as np
import pandas as pd
import pandas.testing as pdt
import toolkit as ftk


class TestData(unittest.TestCase):

    def test_yahoo(self):
        self.assertAlmostEqual(ftk.get_yahoo('^SPX')['2025-12-31'], 6845.50, 2)
        pdt.assert_series_equal(ftk.get_yahoo_bulk(['AAPL', 'MSFT']).loc['2025-12-31', :],
                                pd.Series([271.86, 483.62],
                                          index=['AAPL', 'MSFT']),
                                check_names=False)

    def test_famafrench(self):
        ds = ftk.get_famafrench_datasets()
        self.assertEqual(len(ds), 25)
        # Asia_Pacific_ex_Japan_3_Factors
        pdt.assert_series_equal(ftk.get_famafrench_factors(ds[0]).loc['2025-12'],
                                pd.Series([1.28, 2.56, 0.24, 0.34], index=[
                                          'Mkt-RF', 'SMB', 'HML', 'RF']) / 100,
                                check_names=False)

    def test_msci(self):
        p = ftk.get_msci([892400], ror=False)
        r = ftk.get_msci([892400], ror=True)
        # MSCI ACWI Index Price USD
        self.assertAlmostEqual(p.loc['2025-12-31'].iloc[0], 1014.62, 2)
        self.assertAlmostEqual(
            np.expm1(np.log1p(r.loc['2025']).sum()).iloc[0], 0.2060, 4)
