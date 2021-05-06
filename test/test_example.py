import numpy as np
import unittest

class TestExample(unittest.TestCase): 
    # We use a class so we can use special asserts like self.assertEqual or assertIsInstance, but just the simple builtin "assert is also good"
    def test_sum(self):
        self.assertEqual(2+2,4, "The sum doesn't work")

# This is not defined within a class so unittest doesn't recognize it as a test. Other testing frameworks, like pytest (see https://docs.pytest.org/en/6.2.x/) have an 
# advantage in this regard, but unittest is very standard
def test_product():
    # we cannot use things like assertEqual if we are not inside a unittest.TestCase
    assert 4*5 == 20, "Product doesn't work"