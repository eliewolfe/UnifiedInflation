import numpy as np
import unittest

""" 
The tests in this file are recognized because it begins with "test" (depends on how you configured your environment).
"""

def myfunction(a,b):
    return np.kron(a,b)

class Testmyfunction(unittest.TestCase):
    def test_myfunction(self):
        out_fun = myfunction(np.eye(2),np.eye(2))
        out_correct = np.eye(4)
        self.assertEqual(np.linalg.norm(out_fun-out_correct)<1e-16,True,"Tensor product of identities is not the identity.")