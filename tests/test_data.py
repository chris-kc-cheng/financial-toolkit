import unittest
import toolkit as ftk

class TestData(unittest.TestCase):

    def test_yahoo(self):
        ftk.get_yahoo('TSLA')
        ftk.get_yahoo_bulk(['AAPL', 'MSFT'])

    def test_famafrench(self):
        ds = ftk.get_famafrench_datasets()
        ftk.get_famafrench_factors(ds[0])

    def test_msci(self):
        acwi = ftk.get_msci([892400])
        self.assertAlmostEqual(acwi.loc['2023-12-29'].iloc[0], 726.996)