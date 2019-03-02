import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


class nodes(object):
    def __init__(self, fea=-1, value=None, right=None, left=None, result=None):
        self.fea = fea
        self.value = value
        self.right = right
        self.left = left
        self.result = result


class DecisionTree(object):
    __dataSet = None
    __labels = None
    __originalData = None
    __nodes = None

    def readData(self, filename):
        fr = open(filename)
        lines = fr.readlines()
        lineOfNumbers = len(lines)
        dataMatrix = np.zeros((lineOfNumbers, 4))
        for i in range(lineOfNumbers):
            line = lines[i].split("\t")
            for j in range(4):
                dataMatrix[i][j] = float(line[j])
        self.__originalData = dataMatrix
        dataSet = np.zeros((lineOfNumbers, 3))
        labels = []
        dataSet = self.__originalData[:, 0:3]
        labels = self.__originalData[:, -1]
        self.__labels = labels
        self.__dataSet = dataSet

        '''the dataset must be with label'''

    def calculateShan(self, dataSet):
        labelFraction = {}
        if len(dataSet) == 0:
            return 0
        for data in dataSet:
            label = data[-1]
            if label not in labelFraction.keys():
                labelFraction[label] = 1
            else:
                labelFraction[label] += 1
        Ent = 0.0
        alldata = len(dataSet)
        for key in labelFraction:
            prob = float(labelFraction[key]) / alldata
            Ent -= prob * math.log(prob, 2)
        return Ent

    def choosetheBestFeature(self, dataset):
        feature_num = dataset.shape[1] - 1
        bestEntDiff = 0.0
        BestFeature = -1
        Bestvalue = -1
        baseEnt = self.calculateShan(dataset)
        aSet = None
        bSet = None
        for i in range(feature_num):
            for value in dataset[:, i]:
                A, B = self.split_data(dataset, i, value)
                proA = len(A) / len(dataset)
                proB = 1 - proA
                infoEnt = proA * self.calculateShan(A) + proB * self.calculateShan(B)
                if baseEnt - infoEnt > bestEntDiff:
                    bestEntDiff = baseEnt - infoEnt
                    BestFeature = i
                    Bestvalue = value
                    aSet = A
                    bSet = B
        return BestFeature, Bestvalue, np.array(aSet), np.array(bSet)

    def split_data(self, dataset, feature, value):
        aSet = []
        bSet = []
        for data in dataset:
            if data[feature] > value:
                aSet.append(data)
            else:
                bSet.append(data)
        return aSet, bSet

    def build_tree(self):
        self.__nodes = self.__build_tree(self.__originalData)
        print("Build tree successfully!")

    def __build_tree(self, dataSet):
        if dataSet is None:
            return None
        elif len(dataSet) < 3:
            label_class = {}
            for data in dataSet:
                label = data[-1]
                if label not in label_class:
                    label_class[label] = 1
                else:
                    label_class[label] += 1
            sorted(label_class.items(), key=lambda label_class: label_class[1], reverse=True)
            key = list(label_class)[0]
            return nodes(result=key)
        node = nodes()
        nodes.result = None
        feature, value, A, B = self.choosetheBestFeature(dataSet)
        if value != -1 and feature != -1:
            node.value = value
            node.fea = feature
            node.left = self.__build_tree(A)
            node.right = self.__build_tree(B)
        else:
            label_class = {}
            for data in dataSet:
                label = data[-1]
                if label not in label_class:
                    label_class[label] = 1
                else:
                    label_class[label] += 1
            sorted(label_class.items(), key=lambda label_class: label_class[1], reverse=True)
            key = list(label_class)[0]
            node.result = key
        return node

    def __predict(self, dataset, nodes):
        root = nodes
        while root.result is None:
            if dataset[nodes.fea] > nodes.value:
                root = root.left
            else:
                root = root.right
        return root.result

    def predictTraining(self):
        grades = 0
        for i in range(len(self.__dataSet)):
            ta = self.__predict(self.__dataSet[i, :], self.__nodes)
            if ta == self.__labels[i]:
                grades += 1
        return grades / len(self.__labels)
