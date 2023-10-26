"""_summary_
"""

import unittest
from toolkit import functional as ftk

class TestFunctional(unittest.TestCase):
    """_summary_

    Args:
        unittest (_type_): _description_
    """

    def test_add(self):
        """_summary_
        """
        self.assertEqual(ftk.add(1, 2), 4)

if __name__ == '__main__':
    unittest.main()
    