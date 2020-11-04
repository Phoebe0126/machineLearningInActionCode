import util
from horseLogReg import HorseLogReg


def colicTest(trainPath, testPath):
    # read data from training dataset and test dataset
    trainingInputs, trainingLabels = util.loadDataSet(trainPath)
    testInputs, testLabels = util.loadDataSet(testPath)

    logRegModel = HorseLogReg(0.01, 500)
    trainTheta = logRegModel.fit(trainingInputs, trainingLabels)
    errorCount = 0
    numTestVec = len(testLabels)
    for i in range(numTestVec):
        if int(util.classifyVector(testInputs[i], trainTheta)) != testLabels[i]:
            errorCount += 1
    errorRate = float(errorCount) / float(numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def mutiTest():
    testNum = 10
    errorRateSum = 0.0

    for i in range(testNum):
        errorRateSum += colicTest('./data/horseColicTraining.txt', './data/horseColicTest.txt')
    print("the error rate after average is: %f" % (errorRateSum / float(testNum)))


if __name__ == '__main__':
    mutiTest()
