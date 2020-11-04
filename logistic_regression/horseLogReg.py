from logRegression import LogRegression
import numpy as np
import util


class HorseLogReg(LogRegression):
    def __init__(self, alpha, maxCycles):
        super().__init__(alpha, maxCycles)

    def fit(self, x, y):
        m, n = np.shape(x)
        thetas = np.ones((n, 1))
        for j in range(self.maxCycles):
            dataIndex = list(range(m))
            for i in range(m):
                # decrease alpha
                self.alpha = 4 / (1.0 + j + i) + 0.0001
                randIndex = int(np.random.uniform(0, len(dataIndex)))
                h = util.sigmoid(sum(x[randIndex] * thetas))
                error = y[randIndex] - h
                thetas += (self.alpha * error * x[randIndex]).T
                del (dataIndex[randIndex])
        return thetas
