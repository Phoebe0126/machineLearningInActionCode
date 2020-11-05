import numpy as np


# 计算y=1时xi的概率和y=0时xi的概率
def train(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # 垃圾邮件占的比例
    pAbusive = sum(trainCategory) / float(numTrainDocs)

    # 每个词都+1 平滑
    # 正常邮件各词的数量
    p0Num = np.ones(numWords)
    # 垃圾邮件各词占的数量
    p1Num = np.ones(numWords)

    # 平滑 +2 k=2,因为label有两个值: 1和0
    # 分别代表垃圾邮件或者正常邮件出现的总词数
    p0Denom = 2.0
    p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # 避免两个很小的数相乘溢出
    p1Vector = np.log(p1Num / p1Denom)
    p0Vector = np.log(p0Num / p0Denom)

    return p0Vector, p1Vector, pAbusive


# 计算给定x，y=1的概率
def classifyNB(vec2Classify, p0Vector, p1Vector, pClass1):
    # 注意式子整体log
    # 数组对应位置相乘
    p1 = sum(vec2Classify * p1Vector) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vector) + np.log(1.0 - pClass1)
    # 分母相同，只比较分子即可
    if p1 > p0:
        return 1
    else:
        return 0


