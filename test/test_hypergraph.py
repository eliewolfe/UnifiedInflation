import unittest

# from sdp_utils import solveSDP
from quantuminflation.general_tools import *

class TestHypergraphUtils(unittest.TestCase):
    def test_hypergraphs_are_equal(self):
        hyper1 = np.array([[0, 1, 1],
                           [1, 1, 0],
                           [1, 0, 1],
                           [1, 0, 1]])

        hyper2 = np.array([[0, 1, 1],
                           [1, 1, 0],
                           [1, 0, 1]])

        hyper3 = np.array([[0, 1, 1],
                           [1, 0, 1]])

        hyper4 = np.array([[0, 1, 1],
                           [1, 0, 1],
                           [1, 1, 0]])

        assert hypergraphs_are_equal(hyper1, hyper2), "Incorrect treatment of duplicate hyperlinks"
        assert not hypergraphs_are_equal(hyper2, hyper3), "Obviously incorrect"
        assert hypergraphs_are_equal(hyper2, hyper4), "Does not detect permutations of list of hyperlinks"
