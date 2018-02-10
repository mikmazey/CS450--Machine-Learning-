from __future__ import print_function
import numpy as np
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import math
import random
import operator
from math import log



#categorical = classes
#columns = features

import pandas
import numpy as Np
import scipy.stats as st

from sklearn.model_selection import train_test_split

from math import log
import operator


lenses_data = pandas.read_csv("/Users/darrenfox/PycharmProjects/arcade/data/data_for_CS450/lenses.csv")


columns = ["Age of Patient", "Spectacle Prescription", "Astigmatic", "Tear Production Rate", "Contacts"]

lenses_data.columns = columns


lenses = lenses_data
#make dataset without target in it- will be original dataset without target
lenses_new = lenses[["Age of Patient", "Spectacle Prescription", "Astigmatic", "Tear Production Rate"]].copy()

#print(lenses_new)
#making target data
t = lenses_data
#our dataframe for target is called target lol
#target = t[["Contacts"]].copy()



#print(target)

#x_train, y_train, x_test, y_test = train_test_split(lenses_new, target, test_size=0.3, train_size=0.7)
#print(x_train, y_train)
#print(x_test, y_test)

#train_data = x_train, y_train
#print(train_data)
#test_data = x_test, y_test
#print(test_data)


def calcShannonEnt(lenses_data):
    numEntries = len(lenses_data)
    labelCounts = {}
    for featVec in lenses_data:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
            prob = float(labelCounts[key]) / numEntries
            shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def splitDataSet(self,lenses_data, axis, value):
    retDataSet = []

    for featVec in lenses_data:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet



def chooseBestFeatureToSplit(lenses_data):
    numFeatures = len(lenses_data[0]) -1 #last column is for the labels
    baseEntropy = calcShannonEnt(lenses_data)
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):
            featList = [lenses_new[i] for lenses_new in lenses_data]
            uniqueVals = set(featList)
            newEntropy = 0.0

            for value in uniqueVals:
                subDataSet = splitDataSet(lenses_data, i, value)
                prob = len(subDataSet) / float(len(lenses_data))
                newEntropy += prob * calcShannonEnt(subDataSet)


            infoGain = baseEntropy - newEntropy

            if (infoGain > bestInfoGain):
                bestInfoGain = infoGain
                bestFeature = i

    return bestFeature

def majorityCnt(contact_list):
    contactCount = {}
    for vote in contact_list:
        if vote not in contactCount.keys():
            contactCount[vote] =0
        contactCount[vote] += 1
    sortedContactCount = sorted(contactCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedContactCount[0][0]


def createTree(lenses_data, lenses_new):
    contact_list = [lenses_new[-1] for column in lenses_data]
    if contact_list.count(contact_list[0]) == len(contact_list):
        return contact_list[0] #stop splitting when columns are equal
    if len(lenses_data[0]) ==1:
        return majorityCnt(contact_list)

    #Use information gain
    bestCol = chooseBestFeatureToSplit(lenses_data)
    bestColLabel = lenses_new[bestCol]

    #building tree recursively
    myTree = {bestColLabel: {}}
    del (lenses_new[bestCol])
    labelValues = [lenses_new[bestCol] for lenses_new in lenses_data]
    uniqueVals = set(labelValues)
    for value in uniqueVals:
        subLabels = lenses_new[:] #copy all the labels so trees don't mess up exhisting labels
        myTree[bestColLabel][value] = createTree(train_test_split(lenses_data, bestCol, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]

    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)

    else:
        classLabel = valueOfFeat

    return classLabel

def storeTree(inputTree):
    import pickle
    lenses_data = pandas.read_csv("/Users/darrenfox/PycharmProjects/arcade/data/data_for_CS450/lenses.csv")
    pickle.dump(inputTree, lenses_data)
    lenses_data.close()

def grabTree(lenses_data):
    import pickle
    fr = pandas.read_csv("/Users/darrenfox/PycharmProjects/arcade/data/data_for_CS450/lenses.csv")
    return pickle.load(fr)

#build tree

mytree = createTree(lenses_data, lenses_new)
print(mytree)





