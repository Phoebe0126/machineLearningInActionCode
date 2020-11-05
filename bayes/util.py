import re
import numpy as np


# split big string input into word list
def textParse(bigString):
    listOfToken = re.split(r'\W+', bigString)
    # prevent url like www.abc.com?en=cn&id=aa
    return [token.lower() for token in listOfToken if len(token) > 2]


# get union of the given two sets
def createVocabList(docList):
    docSet = set([])
    for document in docList:
        docSet = docSet | set(document)
    return list(docSet)


# create word vector
def toVecMN(vocabList, inputSet):
    resultVec = np.zeros(len(vocabList))
    for word in inputSet:
        if word in vocabList:
            resultVec[vocabList.index(word)] += 1
    return resultVec


def loadText():
    docList = []
    classList = []
    fullText = []

    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i, encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullText.extend(wordList)
        # spam email classify as 1
        classList.append(1)

        wordList = textParse(open('email/ham/%d.txt' % i, encoding="ISO-8859-1").read())
        docList.append(wordList)
        fullText.extend(wordList)
        # ham email classify as 0
        classList.append(0)

    # create vocabulary
    vocabList = createVocabList(docList)

    # create test set
    trainingSet = list(range(50))
    testList = []
    testClasses = []
    # get test set randomly
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testClasses.append(classList[trainingSet[randIndex]])
        testList.append(toVecMN(vocabList, docList[trainingSet[randIndex]]))
        del(trainingSet[randIndex])

    trainMat = []
    trainClasses = []
    # get word vector list for the selected train set(len:50)
    for docIndex in trainingSet:
        trainMat.append(toVecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    return trainMat, trainClasses, testList, testClasses



