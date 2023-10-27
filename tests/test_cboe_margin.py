import unittest
import pandas as pd
from toolkit.cboe_margin import margin, margin_strategy, max_loss


class TestMargin(unittest.TestCase):

    def test_long_options_9_months_or_less(self):
        # 23
        self.assertAlmostEqual(margin(
            otype='c', quantity=1, expiration=6/12, strike=125, value=5, underlying=128.5), 500)
        self.assertAlmostEqual(margin(
            otype='p', quantity=1, expiration=6/12, strike=430, value=5.5, underlying=433.35), 550)

    def test_long_options_more_than_9_months(self):
        # 24-25
        self.assertAlmostEqual(margin(
            otype='c', quantity=1, expiration=18/12, strike=80, value=12, underlying=78), 900)
        self.assertAlmostEqual(margin(
            otype='c', quantity=1, expiration=20/12, strike=1325, value=16.8, underlying=1290), 1260)
        self.assertAlmostEqual(
            margin(otype='p', quantity=1, expiration=2, strike=42.5, value=2), 150)

    def test_long_options_otc_more_than_9_months(self):
        # 26-27
        self.assertAlmostEqual(margin(otype='c', quantity=1, expiration=1,
                               strike=75, otc=True, value=4.5, underlying=79), 350)
        self.assertAlmostEqual(margin(otype='c', quantity=1, expiration=1,
                               strike=665, otc=True, value=11, underlying=667.34), 1041.5)
        self.assertAlmostEqual(margin(otype='c', quantity=1, expiration=1,
                               strike=665, otc=True, value=13, underlying=663.5), 1300)

    def test_short_put(self):
        # 28
        self.assertAlmostEqual(
            margin(otype='p', quantity=-1, strike=80, value=2, underlying=95), 1000)
        self.assertAlmostEqual(
            margin(otype='p', quantity=-1, strike=20, value=1.5, underlying=19.5), 540)

    def test_short_put_leveraged(self):
        # 29
        self.assertAlmostEqual(margin(
            otype='p', quantity=-1, strike=725, leverage=2, value=3, underlying=970), 14800)
        self.assertAlmostEqual(margin(
            otype='p', quantity=-1, strike=400, leverage=3, value=15.5, underlying=390.7), 24992)

    def test_short_put_broad_based(self):
        # 30-31
        self.assertAlmostEqual(margin(
            otype='p', quantity=-1, strike=410, broad=True, value=0.1, underlying=445.35), 4110)
        self.assertAlmostEqual(margin(
            otype='p', quantity=-1, strike=430, broad=True, value=7.8, underlying=433.35), 6945.25)
        self.assertAlmostEqual(margin(
            otype='p', quantity=-1, strike=45, broad=True, value=2.85, underlying=43.34), 935.1)
        self.assertAlmostEqual(margin(otype='p', quantity=-1, strike=390,
                               broad=True, leverage=1.5, value=4, underlying=460), 6250)

    def test_short_call(self):
        # 32-35
        self.assertAlmostEqual(
            margin(otype='c', quantity=-1, strike=120, value=8.4, underlying=128.5), 3410)
        self.assertAlmostEqual(
            margin(otype='c', quantity=-1, strike=30, value=0.05, underlying=26.38), 268.8)
        self.assertAlmostEqual(margin(
            otype='c', quantity=-1, strike=810, leverage=2, value=10.1, underlying=815.5), 33630)
        self.assertAlmostEqual(margin(
            otype='c', quantity=-1, strike=1250, leverage=1.5, value=2, underlying=1050.3), 15954.5)
        self.assertAlmostEqual(margin(
            otype='c', quantity=-1, strike=430, broad=True, value=8.7, underlying=433.35), 7370.25)
        self.assertAlmostEqual(margin(
            otype='c', quantity=-1, strike=45, broad=True, value=1.35, underlying=43.34), 619.1)
        self.assertAlmostEqual(margin(
            otype='c', quantity=-1, strike=370, broad=True, value=12.85, underlying=378.5), 6962.5)
        self.assertAlmostEqual(margin(otype='c', quantity=-1, strike=450,
                               broad=True, leverage=1.5, value=3, underlying=410), 6450)

    def test_max_loss(self):
        p1 = pd.DataFrame([{'otype': 'c', 'quantity': 10, 'strike': 75, 'multiplier': 100, 'reduced': 1},
                           {'otype': 'c', 'quantity': -10, 'strike': 85, 'multiplier': 100, 'reduced': 1}])

        p2 = pd.DataFrame([{'otype': 'c', 'quantity': 10, 'strike': 45, 'multiplier': 100, 'reduced': 1},
                           {'otype': 'c', 'quantity': -20, 'strike': 55,
                               'multiplier': 100, 'reduced': 1},
                           {'otype': 'c', 'quantity': 10, 'strike': 60, 'multiplier': 100, 'reduced': 1}])

        p3 = pd.DataFrame([{'otype': 'p', 'quantity': 5, 'strike': 60, 'multiplier': 100, 'reduced': 1},
                           {'otype': 'p', 'quantity': -5, 'strike': 70,
                               'multiplier': 100, 'reduced': 1},
                           {'otype': 'c', 'quantity': -10, 'strike': 85,
                               'multiplier': 100, 'reduced': 1},
                           {'otype': 'c', 'quantity': 10, 'strike': 90, 'multiplier': 100, 'reduced': 1}])

        p4 = pd.DataFrame([{'otype': 'c', 'quantity': 10, 'strike': 45, 'multiplier': 100, 'reduced': 1},
                           {'otype': 'c', 'quantity': -10, 'strike': 55,
                               'multiplier': 100, 'reduced': 1},
                           {'otype': 'c', 'quantity': -10, 'strike': 55,
                               'multiplier': 100, 'reduced': 1},
                           {'otype': 'c', 'quantity': 10, 'strike': 60, 'multiplier': 100, 'reduced': 1}])
        # 36-38
        self.assertAlmostEqual(max_loss(p1), 0)
        self.assertAlmostEqual(max_loss(p2), 0)
        self.assertAlmostEqual(max_loss(p3), 5000)
        self.assertAlmostEqual(max_loss(p4), 0)

    def test_put_spread(self):
        p1 = pd.DataFrame([{'otype': 'p', 'quantity': 1, 'strike': 250, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 3},
                           {'otype': 'p', 'quantity': -1, 'strike': 240, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 0.95}])
        p2 = pd.DataFrame([{'otype': 'p', 'quantity': 1, 'strike': 425, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 6.4},
                           {'otype': 'p', 'quantity': -1, 'strike': 430, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 7.8}])
        p3 = pd.DataFrame([{'otype': 'p', 'quantity': 1, 'strike': 42.5, 'expiration': 18/12, 'multiplier': 100, 'reduced': 1/10, 'value': 2},
                           {'otype': 'p', 'quantity': -1, 'strike': 45, 'expiration': 18/12, 'multiplier': 100, 'reduced': 1/10, 'value': 2.9}])
        p4 = pd.DataFrame([{'otype': 'p', 'quantity': 1, 'strike': 430, 'expiration': 5/12, 'multiplier': 100, 'reduced': 1, 'broad': True, 'value': 7.9},
                           {'otype': 'p', 'quantity': -10, 'strike': 42.5, 'expiration': 18/12, 'multiplier': 100, 'reduced': 1/10, 'broad': True, 'value': 2}])
        # 39-41
        self.assertAlmostEqual(margin_strategy(p1, 255), 300)
        self.assertAlmostEqual(margin_strategy(p2, 433.35), 1140)
        self.assertAlmostEqual(margin_strategy(p3, 433.4), 450)
        self.assertAlmostEqual(margin_strategy(p4, 433.4), 8451)

    def test_call_spread(self):
        p1 = pd.DataFrame([{'otype': 'c', 'quantity': 1, 'strike': 125, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 3.8},
                           {'otype': 'c', 'quantity': -1, 'strike': 120, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 8.4}])
        p2 = pd.DataFrame([{'otype': 'c', 'quantity': 1, 'strike': 70, 'expiration': 9/12, 'multiplier': 100, 'reduced': 1, 'value': 5},
                           {'otype': 'c', 'quantity': -1, 'strike': 70, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 8}])
        p3 = pd.DataFrame([{'otype': 'c', 'quantity': 1, 'strike': 425, 'expiration': 11/12, 'multiplier': 100, 'reduced': 1, 'broad': True, 'value': 13.1},
                           {'otype': 'c', 'quantity': -1, 'strike': 430, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'broad': True, 'value': 12.2}])
        p4 = pd.DataFrame([{'otype': 'c', 'quantity': 10, 'strike': 45, 'expiration': 18/12, 'multiplier': 100, 'reduced': 1/10, 'broad': True, 'value': 1.35},
                           {'otype': 'c', 'quantity': -1, 'strike': 450, 'expiration': 3/12, 'multiplier': 100, 'reduced': 1, 'broad': True, 'value': 0.25}])
        # 42-44
        self.assertAlmostEqual(margin_strategy(p1, 128.5), 880)
        self.assertAlmostEqual(margin_strategy(p2, 75), 2800)
        self.assertAlmostEqual(margin_strategy(p3, 433.35), 9030.25)
        self.assertAlmostEqual(margin_strategy(p4, 433.4), 1350)

    def test_straddle(self):
        p1 = pd.DataFrame([{'otype': 'c', 'quantity': -1, 'strike': 90, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 7},
                           {'otype': 'p', 'quantity': -1, 'strike': 90, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 3.7}])
        p2 = pd.DataFrame([{'otype': 'p', 'quantity': -1, 'strike': 435, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'broad': True, 'value': 7.2},
                           {'otype': 'c', 'quantity': -1, 'strike': 435, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'broad': True, 'value': 5.5}])
        p3 = pd.DataFrame([{'otype': 'p', 'quantity': -1, 'strike': 45, 'expiration': 1, 'multiplier': 100, 'reduced': 1/10, 'broad': True, 'value': 2.85},
                           {'otype': 'c', 'quantity': -1, 'strike': 45, 'expiration': 1, 'multiplier': 100, 'reduced': 1/10, 'broad': True, 'value': 1.35}])
        # 45-46
        self.assertAlmostEqual(margin_strategy(p1, 92.63), 2922.6)
        self.assertAlmostEqual(margin_strategy(p2, 433.35), 7770.25)
        self.assertAlmostEqual(margin_strategy(p3, 433.4), 1070.1)

    def test_underlying_with_short_option(self):
        p1 = pd.DataFrame([{'otype': 'u', 'quantity': -100, 'value': 255},
                           {'otype': 'p', 'quantity': -1, 'strike': 250, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 3}])
        p2 = pd.DataFrame([{'otype': 'u', 'quantity': 100, 'value': 92.38},
                           {'otype': 'c', 'quantity': -1, 'strike': 90, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 7}])
        p3 = pd.DataFrame([{'otype': 'u', 'quantity': 10111, 'value': 36.1},
                           {'otype': 'p', 'quantity': -1, 'strike': 3600, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 45}])
        # 47-48
        self.assertAlmostEqual(margin_strategy(p1), 38250)
        self.assertAlmostEqual(margin_strategy(p2), 4619)
        self.assertAlmostEqual(margin_strategy(p3), 182503.55)

    def test_box_spread(self):
        p1 = pd.DataFrame([{'otype': 'c', 'quantity': 1, 'strike': 535, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 19.4},
                           {'otype': 'c', 'quantity': -1, 'strike': 545, 'expiration': 1,
                               'multiplier': 100, 'reduced': 1, 'value': 12.2},
                           {'otype': 'p', 'quantity': 1, 'strike': 545, 'expiration': 1,
                               'multiplier': 100, 'reduced': 1, 'value': 5.3},
                           {'otype': 'p', 'quantity': -1, 'strike': 535, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 3}])
        p2 = pd.DataFrame([{'otype': 'c', 'quantity': 1, 'strike': 40, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 15.4},
                           {'otype': 'c', 'quantity': -1, 'strike': 50, 'expiration': 1,
                               'multiplier': 100, 'reduced': 1, 'value': 7.2},
                           {'otype': 'p', 'quantity': 1, 'strike': 50, 'expiration': 1,
                               'multiplier': 100, 'reduced': 1, 'value': 1.75},
                           {'otype': 'p', 'quantity': -1, 'strike': 40, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 0.4}])
        p3 = pd.DataFrame([{'otype': 'c', 'quantity': -1, 'strike': 50, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 17.3},
                           {'otype': 'c', 'quantity': 1, 'strike': 60, 'expiration': 1,
                               'multiplier': 100, 'reduced': 1, 'value': 8.4},
                           {'otype': 'p', 'quantity': -1, 'strike': 60, 'expiration': 1,
                               'multiplier': 100, 'reduced': 1, 'value': 1.25},
                           {'otype': 'p', 'quantity': 1, 'strike': 50, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 0.3}])
        # 49-51
        self.assertAlmostEqual(margin_strategy(p1, 550, eligible=True), 500)
        self.assertAlmostEqual(margin_strategy(p2, 50), 1715)
        self.assertAlmostEqual(margin_strategy(p3, 66), 1870)

    def test_long_butterfly_spread(self):
        p1 = pd.DataFrame([{'otype': 'p', 'quantity': 1, 'strike': 540, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 5.6},
                           {'otype': 'p', 'quantity': -2, 'strike': 550, 'expiration': 1,
                               'multiplier': 100, 'reduced': 1, 'value': 7.2},
                           {'otype': 'p', 'quantity': 1, 'strike': 555, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 9.8}])
        p2 = pd.DataFrame([{'otype': 'c', 'quantity': 1, 'strike': 545, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 12.4},
                           {'otype': 'c', 'quantity': -2, 'strike': 550, 'expiration': 1,
                               'multiplier': 100, 'reduced': 1, 'value': 8.8},
                           {'otype': 'c', 'quantity': 1, 'strike': 565, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 2}])
        # 52-53
        self.assertAlmostEqual(margin_strategy(p1, 550), 2040)
        self.assertAlmostEqual(margin_strategy(p2, 550), 2440)

    def test_long_condor_spread(self):
        p1 = pd.DataFrame([{'otype': 'p', 'quantity': 1, 'strike': 1050, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 47.1},
                           {'otype': 'p', 'quantity': -1, 'strike': 1075, 'expiration': 1,
                               'multiplier': 100, 'reduced': 1, 'value': 55.7},
                           {'otype': 'p', 'quantity': -1, 'strike': 1100, 'expiration': 1,
                               'multiplier': 100, 'reduced': 1, 'value': 66.3},
                           {'otype': 'p', 'quantity': 1, 'strike': 1125, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 85.4}])
        p2 = pd.DataFrame([{'otype': 'c', 'quantity': 1, 'strike': 22.5, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 0.45},
                           {'otype': 'c', 'quantity': -1, 'strike': 25, 'expiration': 1,
                               'multiplier': 100, 'reduced': 1, 'value': 0.75},
                           {'otype': 'c', 'quantity': -1, 'strike': 30, 'expiration': 1,
                               'multiplier': 100, 'reduced': 1, 'value': 2.75},
                           {'otype': 'c', 'quantity': 1, 'strike': 32.5, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 4.3}])
        # 54-55
        self.assertAlmostEqual(margin_strategy(p1, 1160), 13250)
        self.assertAlmostEqual(margin_strategy(p2, 26.75), 475)

    def test_short_iron_butterfly_spread(self):
        p1 = pd.DataFrame([{'otype': 'p', 'quantity': 1, 'strike': 16, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 0.1},
                           {'otype': 'p', 'quantity': -1, 'strike': 20, 'expiration': 1,
                               'multiplier': 100, 'reduced': 1, 'value': 0.2},
                           {'otype': 'c', 'quantity': -1, 'strike': 20, 'expiration': 1,
                               'multiplier': 100, 'reduced': 1, 'value': 7},
                           {'otype': 'c', 'quantity': 1, 'strike': 24, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 4}])
        # 56
        self.assertAlmostEqual(margin_strategy(p1, 26.75), 810)

    def test_short_iron_condor_spread(self):
        p1 = pd.DataFrame([{'otype': 'p', 'quantity': 1, 'strike': 1000, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 32},
                           {'otype': 'p', 'quantity': -1, 'strike': 1025, 'expiration': 1,
                               'multiplier': 100, 'reduced': 1, 'value': 35.1},
                           {'otype': 'c', 'quantity': -1, 'strike': 1150, 'expiration': 1,
                               'multiplier': 100, 'reduced': 1, 'value': 9.4},
                           {'otype': 'c', 'quantity': 1, 'strike': 1175, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 6.3}])
        # 57
        self.assertAlmostEqual(margin_strategy(p1, 1060), 6330)

    def test_underlying_with_long_option(self):
        p1 = pd.DataFrame([{'otype': 'u', 'quantity': 100, 'value': 103.5},
                           {'otype': 'p', 'quantity': 1, 'strike': 95, 'expiration': 1, 'multiplier': 100, 'reduced': 1}])
        p2 = pd.DataFrame([{'otype': 'u', 'quantity': -100, 'value': 46},
                           {'otype': 'c', 'quantity': 1, 'strike': 50, 'expiration': 1, 'multiplier': 100, 'reduced': 1}])
        # 58
        self.assertAlmostEqual(margin_strategy(p1, 103.5), 1800)
        self.assertAlmostEqual(margin_strategy(p2, 46), 5900)

    def test_conversion(self):
        p1 = pd.DataFrame([{'otype': 'u', 'quantity': 100, 'value': 115},
                           {'otype': 'c', 'quantity': -1, 'strike': 110, 'expiration': 1,
                               'multiplier': 100, 'reduced': 1, 'value': 6.5},
                           {'otype': 'p', 'quantity': 1, 'strike': 110, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 1.35}])
        p2 = pd.DataFrame([{'otype': 'u', 'quantity': -100, 'value': 115},
                           {'otype': 'c', 'quantity': 1, 'strike': 110, 'expiration': 1,
                               'multiplier': 100, 'reduced': 1, 'value': 6.5},
                           {'otype': 'p', 'quantity': -1, 'strike': 110, 'expiration': 1, 'multiplier': 100, 'reduced': 1, 'value': 1.35}])
        p3 = pd.DataFrame([{'otype': 'u', 'quantity': -100, 'value': 71.9},
                           {'otype': 'c', 'quantity': 1, 'strike': 75,
                               'expiration': 1, 'multiplier': 100, 'reduced': 1},
                           {'otype': 'p', 'quantity': -1, 'strike': 75, 'expiration': 1, 'multiplier': 100, 'reduced': 1}])
        # 59-61
        self.assertAlmostEqual(margin_strategy(p1, 115), 1100)
        self.assertAlmostEqual(margin_strategy(p2, 115), 12100)
        self.assertAlmostEqual(margin_strategy(p3, 71.9), 8560)

    def test_collar(self):
        p1 = pd.DataFrame([{'otype': 'u', 'quantity': 100, 'value': 31.75},
                           {'otype': 'p', 'quantity': 1, 'strike': 30,
                               'expiration': 1, 'multiplier': 100, 'reduced': 1},
                           {'otype': 'c', 'quantity': -1, 'strike': 35, 'expiration': 1, 'multiplier': 100, 'reduced': 1}])
        # 61
        self.assertAlmostEqual(margin_strategy(p1, 31.75), 475)
