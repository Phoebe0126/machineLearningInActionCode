import bayes.util as util
import bayes.bayesModel as bayesModel


def spamTest():
    trainMat, trainClasses, testList, testClasses = util.loadText()
    p0Vector, p1Vector, p1Class = bayesModel.train(trainMat, trainClasses)

    # test start
    i = 0
    errorCount = 0
    while i < len(testClasses):
        testResult = bayesModel.classifyNB(testList[i], p0Vector, p1Vector, p1Class)
        if testResult != testClasses[i]:
            errorCount += 1
        i += 1
    print('the error rate is:', float(errorCount) / len(testClasses))


if __name__ == '__main__':
    spamTest()
