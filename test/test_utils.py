import time
import numpy as np
import scipy as sp
from scipy import stats
import unittest
import pytest

from test_context import hpt, knockpy, utilities

class TestMetrics(unittest.TestCase):
    """ Tests sparse bayesian linear reg """

    def test_count_randomized_pairs(self):

        # Simple test case #1
        rand_pairs, rand_singletons = utilities._count_rand_pairs_inner(
            nonints=[0.55, 0.45, 0.123], 
            groups=[set([1]), set([1,2]), set([3])]
        )
        self.assertTrue(
            rand_pairs == [(0,1)],
            "count randomized pairs yields wrong randomized pairs."
        )
        self.assertTrue(
            rand_singletons == set([2]),
            "count randomized pairs yields wrong singletons"
        )

        # Simple test case #2
        nonints = [0.95, 0.05, 0.02, 0.01, 0.98, 0.99]
        groups = [set([1]), set([1,2]), set([3,4]), set([5,6]), set([3]), set([4])]
        rand_pairs, rand_singletons = utilities._count_rand_pairs_inner(
            nonints=nonints, groups=groups
        )
        self.assertTrue(
            rand_singletons == set([3,5]),
            "count randomized pairs yields wrong singletons"
        )





if __name__ == "__main__":
    unittest.main()
