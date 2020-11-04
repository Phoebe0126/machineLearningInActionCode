import numpy as np
import util


class LogRegression(object):

    def __init__(self, alpha=0.001, maxCycles=500):
        self.alpha = alpha
        self.maxCycles = maxCycles

    def fit(self, x, y):
        """
        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
            @return: theta matrix
        """
        # convert arr to numpy matrix
        x = np.mat(x)
        y = np.mat(y).transpose()

        # get rows and cols of inputs
        m, n = np.shape(x)

        # init theta arr
        thetas = np.zeros((n, 1))

        # iterate for maxCycles times
        for k in range(self.maxCycles):
            h = util.sigmoid(x * thetas)
            error = y - h
            thetas += self.alpha * x.T * error
        return thetas
