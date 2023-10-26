
import unittest
import pandas as pd
import toolkit as ftk

class TestFunctional(unittest.TestCase):
    """_summary_

    Args:
        unittest (_type_): _description_
    """

    def test_compound_return(self):
        """_summary_
        """
        self.assertEqual(ftk.compound_return(pd.Series()), 123)

if __name__ == '__main__':
    unittest.main()
    