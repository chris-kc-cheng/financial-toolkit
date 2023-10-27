import unittest
import toolkit as ftk

class TestGreeks(unittest.TestCase):
    
    def test_price(self):
        self.assertAlmostEqual(ftk.price_call(50, 49, 0.05, 0.3846, 0.2, 0), 2.4, 1)
        self.assertAlmostEqual(ftk.price_call(60, 62, 0.1, 5 / 12, 0.2, 0), 5.798, 3)
        self.assertAlmostEqual(ftk.price_call(40, 43, 0.1, 0.5, 0.2, 0), 5.56, 2)
        self.assertAlmostEqual(ftk.price_call(40, 44, 0.1, 0.5 - 1 / 26, 0.2, 0), 6.23, 2)
        self.assertAlmostEqual(ftk.price_put(60, 65, 0.05, 0.5, 0.2, 0), 1.1567, 4)
    
    def test_delta(self):
        self.assertAlmostEqual(ftk.delta_call(50, 49, 0.05, 0.3846, 0.2, 0), 0.522, 3)
        self.assertAlmostEqual(ftk.delta_call(60, 62, 0.1, 5 / 12, 0.2, 0), 0.739332, 6)
        self.assertAlmostEqual(ftk.delta_call(40, 43, 0.1, 0.5, 0.2, 0), 0.825, 3)
        self.assertAlmostEqual(ftk.delta_put(60, 65, 0.05, 0.5, 0.2, 0), -0.2080, 4)
        
    def test_theta(self):
        self.assertAlmostEqual(ftk.theta_call(50, 49, 0.05, 0.3846, 0.2, 0), -4.31, 2)
        self.assertAlmostEqual(ftk.theta_put(60, 65, 0.05, 0.5, 0.2, 0), -1.90, 2)
        
    def test_gamma(self):
        self.assertAlmostEqual(ftk.gamma(50, 49, 0.05, 0.3846, 0.2, 0), 0.066, 3)

    def test_vega(self):
        self.assertAlmostEqual(ftk.vega(50, 49, 0.05, 0.3846, 0.2, 0), 12.1, 1)

    def test_rho(self):
        self.assertAlmostEqual(ftk.rho_call(50, 49, 0.05, 0.3846, 0.2, 0), 8.91, 2)
        self.assertAlmostEqual(ftk.rho_put(60, 65, 0.05, 0.5, 0.2, 0), -7.34, 2)
    
    def test_put_call_parity(self):
        value = 49
        value -= ftk.price_call(50, 49, 0.05, 0.3846, 0.2, 0)
        value += ftk.price_put(50, 49, 0.05, 0.3846, 0.2, 0)
        value -= ftk.bp(50, 0.05, 0.3846)
        self.assertAlmostEqual(value, 0)