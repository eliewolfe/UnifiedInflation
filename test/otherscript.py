import numpy as np
import unittest

""" 
The tests in this file are not recognized because the .py doesn't start with 'test'.
"""

def myfunction(a,b):
    return np.kron(a,b)

class Testmyfunction(unittest.TestCase):
    def test_myfunction(self):
        out_fun = myfunction(np.eye(2),np.eye(2))
        out_correct = np.eye(4)
        self.assertEqual(out_fun,out_correct,"Tensor product of identities is not the identity.")