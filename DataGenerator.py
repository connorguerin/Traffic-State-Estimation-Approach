import numpy as np
class DataGenerator:
    def __init__(self, n, p):
        self.n = n
        self.p = p

    def generatePoints(self):
        betaTrue = np.random.rand(self.p, 1)
        X = np.random.random_sample((self.n, self.p))*10
        #TODO: define standard deviation?
        error = np.random.randn(self.n, 1)
        Y = np.dot(X, betaTrue) + error
        return X, Y, betaTrue




