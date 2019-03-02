from numpy import *
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt


class KNN(object):
    @staticmethod
    def classify(inX, dataSet, labels, k):
        datasetsize = dataSet.shape[0]
        diffmat = tile(inX, (datasetsize, 1)) - dataSet
        sqdiffMat = diffmat ** 2
        sqdistance = sqdiffMat.sum(axis=1)
        distance = sqdistance ** 0.5
        sortDistance = distance.argsort()
        classcount = {}
        for i in range(k):
            voteLabel = labels[sortDistance[i]]
            classcount[voteLabel] = classcount.get(voteLabel, 0) + 1
        sortclasscount = sorted(iter(classcount), reverse=True)
        return sortclasscount[0]

    @staticmethod
    def filematrix(filename):
        fr = open(filename)
        arrayOlines = fr.readlines()
        numberOfLines = len(arrayOlines)
        returnMat = np.zeros((numberOfLines, 3))
        classLabelVector = []
        index = 0
        for line in arrayOlines:
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index, :] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
        returnMat, a, b = KNN.autoNorm(returnMat)
        return returnMat, classLabelVector

    @staticmethod
    def drawDataSet(X, Y):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(X[:, 0], X[:, 1], 15.0 * np.array(Y), 15.0 * np.array(Y))
        plt.show()

    @staticmethod
    def autoNorm(dataSet):
        minVals = dataSet.min(0)
        maxVals = dataSet.max(0)
        ranges = maxVals - minVals
        normDataset = np.zeros(shape(dataSet))
        m = dataSet.shape[0]
        normDataset = dataSet - tile(minVals, (m, 1))
        normDataset = normDataset / tile(ranges, (m, 1))
        return normDataset, ranges, minVals

    @staticmethod
    def datingClassTest():
        hoRatio = 0.8
        datingDataMat, datingLabels = KNN.filematrix('dataSet/datingTestSet2.txt')  # load data setfrom file
        normMat, ranges, minVals = KNN.autoNorm(datingDataMat)
        m = normMat.shape[0]
        numTestVecs = int(m * hoRatio)
        errorCount = 0.0
        for i in range(numTestVecs):
            classifierResult = KNN.classify(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
            print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
            if (classifierResult != datingLabels[i]): errorCount += 1.0
        print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
        print(errorCount)

    @staticmethod
    def img2vector(filename):
        returnVector = np.zeros((1, 1024))
        fr = open(filename)
        for i in range(32):
            line = fr.readline()
            for j in range(32):
                returnVector[0, 32 * i + j] = int(line[j])
        return returnVector

    @staticmethod
    def generateDigitDataSet():
        end_num = 179
        current_digit = 0
        ditionary = "dataSet/trainingDigits/"
        dataSet_forDigit = np.zeros((end_num * 10, 1024))
        labels = []
        for i in range(10):
            for j in range(179):
                vector = KNN.img2vector(ditionary + str(i) + "_" + str(j) + ".txt")
                dataSet_forDigit[current_digit, :] = vector
                labels.append(i)
                current_digit += 1
        return dataSet_forDigit, labels

    @staticmethod
    def testForDigit(dataSet, label):
        grades = 0
        all = 0
        for i in range(dataSet.shape[0]):
            print(str(i) + " number")
            num = KNN.classify(dataSet[i, :], dataSet, label, 4)
            all += 1
            if num == label[i]:
                grades += 1
        print(grades / all)
