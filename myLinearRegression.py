from math import exp
from math import log2
from numpy.linalg import inv
import numpy as np


# from random import shuffle, random
# from logisticRegression import SGDLogisticTool, myLogisticRegression
from inversion import invertmatrix, my_invert


def identity(N):
    M = [[0 for x in range(N)] for y in range(N)]
    for i in range(0, N):
        M[i][i] = 1
    return M


# functions for multiplying two matrices
def mymullines(v1, v2):
    return sum([x * y for x, y in zip(v1, v2)])


def mymatmulvec(M, v):
    return [mymullines(r, v) for r in M]


def mymatmul(X, Y):
    result = [[sum(x * y for x, y in zip(rowx, coly)) for coly in transpose(Y)] for rowx in X]
    return result


# function for transpose

def transpose(M):
    transp = [[[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]]
    return transp[0]


class MyLinearMultivariateRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    # learn a linear multivariate regression model by using training inputs (x) and outputs (y)
    def fit(self, x, y):
        X = [[1] for i in range(len(x))]
        # firstLineM = np.array(firstLine)
        for i in range(len(x)):
            for j in range(len(x[i])):
                X[i].append(x[i][j])

        # X = np.insert(np.array(x), 0, [1.0], 1)
        self.coef_ = [0.0 for i in range(len(x[1]))]
        #  using formula: BETA = Inversa( Transpus(X) * X ) * Transpus(X) * Y
        transpX = transpose(X)
        B = mymatmulvec(my_invert(mymatmul(transpX, X)), mymatmulvec(transpX, y))
        # B1 = np.matmul(inv(np.matmul(transpX,X)), np.matmul(transpX,y))

        self.intercept_ = B[0]
        self.coef_ = B[1:]

    # predict the outputs for some new inputs (by using the learnt model)
    def predict(self, x):
        if isinstance(x[0], list):
            return [self.intercept_ + self.coef_[0] * val[0] + self.coef_[1] * val[1] for val in x]
        else:
            return [self.intercept_ + self.coef_ * val for val in x]
