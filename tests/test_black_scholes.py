import unittest
import numpy as np
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
    
    def test_put_call_parity_price(self):
        value = 49
        value -= ftk.price_call(50, 49, 0.05, 0.3846, 0.2, 0)
        value += ftk.price_put(50, 49, 0.05, 0.3846, 0.2, 0)
        value -= ftk.bp(50, 0.05, 0.3846)
        self.assertAlmostEqual(value, 0)

    def test_put_call_parity_price_vector(self):
        spot = np.linspace(0, 100, 11)        
        
        value = spot * np.exp(-0.1 * 0.3846)
        value -= ftk.price_call(50, spot, 0.05, 0.3846, 0.2, 0.1)
        value += ftk.price_put(50, spot, 0.05, 0.3846, 0.2, 0.1)
        value -= ftk.bp(50, 0.05, 0.3846)
        np.testing.assert_allclose(value, 0, atol=1e-07)

        delta = ftk.delta_put(50, spot, 0.05, 0.3846, 0.2, 0.1)
        delta -= ftk.delta_call(50, spot, 0.05, 0.3846, 0.2, 0.1)
        delta += np.exp(-0.1 * 0.3846)
        np.testing.assert_allclose(delta, 0, atol=1e-07)

        theta = ftk.theta_put(50, spot, 0.05, 0.3846, 0.2, 0.1)
        theta -= ftk.theta_call(50, spot, 0.05, 0.3846, 0.2, 0.1)
        theta += 0.1 * spot * np.exp(-0.1 * 0.3846)
        theta -= 0.05 * 50 * np.exp(-0.05 * 0.3846)
        np.testing.assert_allclose(theta, 0, atol=1e-07)        

        rho = ftk.rho_put(50, spot, 0.05, 0.3846, 0.2, 0.1)
        rho -= ftk.rho_call(50, spot, 0.05, 0.3846, 0.2, 0.1)
        rho += 50 * 0.3846 * np.exp(-0.05 * 0.3846)
        np.testing.assert_allclose(rho, 0, atol=1e-07)

    def test_approximation(self):
        self.assertAlmostEqual(ftk.delta_call(40, 43, 0.1, 0.5, 0.2, 0) * 1
                               + 0.5 * ftk.gamma(40, 43, 0.1, 0.5, 0.2, 0) * 1 ** 2
                               + ftk.theta_call(40, 43, 0.1, 0.5, 0.2, 0) * 1 / 26,
                               ftk.price_call(40, 44, 0.1, 0.5 - 1 / 26, 0.2, 0)                               
                               - ftk.price_call(40, 43, 0.1, 0.5, 0.2, 0), 2)
        
        self.assertAlmostEqual(ftk.delta_put(40, 43, 0.1, 0.5, 0.2, 0) * 1
                               + 0.5 * ftk.gamma(40, 43, 0.1, 0.5, 0.2, 0) * 1 ** 2
                               + ftk.theta_put(40, 43, 0.1, 0.5, 0.2, 0) * 1 / 26,
                               ftk.price_put(40, 44, 0.1, 0.5 - 1 / 26, 0.2, 0)                               
                               - ftk.price_put(40, 43, 0.1, 0.5, 0.2, 0), 2)