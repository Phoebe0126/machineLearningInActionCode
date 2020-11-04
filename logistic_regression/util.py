import numpy as np
import matplotlib.pyplot as plt


# return sigma function result
def sigmoid(z):
    if z >= 0:
        return 1.0 / (1 + np.exp(-z))
    else:
        return np.exp(z) / (1 + np.exp(z))


def plotFit(x, y, thetas):
    # n: types of features
    n = np.shape(x)[0]
    xcord1 = [];
    ycord1 = []
    xcord2 = [];
    ycord2 = []
    for i in range(n):
        if int(y[i]) == 1:
            xcord1.append(x[i, 1])
            ycord1.append(x[i, 2])
        else:
            xcord2.append(x[i, 1])
            ycord2.append(x[i, 2])

    # init
    fig = plt.figure()
    # # params: rows，cols，index
    # ax = fig.add_subplot(111)
    # ax.scatter(xcord1, ycord1, s=30, c='red', marker='o')
    # ax.scatter(xcord2, ycord2, s=30, c='green', marker='x')

    plt.plot(xcord1, ycord1, 'go', linewidth=2)
    plt.plot(xcord2, ycord2, 'rx', linewidth=2)
    # draw line
    margin1 = (max(x[:, -2]) - min(x[:, -2])) * 0.1

    lineX1 = np.arange(min(x[:, -2]) - margin1, max(x[:, -2]) + margin1, 0.1)
    lineX2 = -((thetas[0] + thetas[1] * lineX1) / thetas[2])
    plt.plot(lineX1, lineX2)

    # set label names
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


# load data from given path
def loadDataSet(loadPath):
    dataMat = []
    labelMat = []
    fr = open(loadPath)

    for line in fr.readlines():
        lineArr = line.strip().split()
        # add default theta0: 1.0
        xLine = [1.0]
        xInput = [float(num) for num in lineArr[0:-1]]
        xLine = np.concatenate((xLine, xInput), axis=0)
        dataMat.append(xLine)
        labelMat.append(int(float(lineArr[-1])))
    return np.mat(dataMat), labelMat


# classify result between 0 and 1
def classifyVector(inX, thetas):
    prob = sigmoid(sum(inX * thetas))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0
