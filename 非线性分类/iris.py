import numpy as np
import pandas as pd
import operator
from sklearn.model_selection import train_test_split

def classToNum(labels):
    tuple = {label for label in labels}
    number = [i for i in range(len(tuple))]
    classCount = dict(zip(tuple, number))
    y = np.array([classCount[label] for label in labels])
    return y

def calculate_Ndistance(inX, dataSet):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqdiffMat = diffMat ** 2
    sqdistances = sqdiffMat.sum(axis=1)
    distances = sqdistances ** 0.5
    return distances

def autoNorm(dataSet):
    m = dataSet.shape[0]
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normdataSet = np.zeros(np.shape(dataSet))
    normdataSet = (dataSet - np.tile(minVals, (m, 1)) / np.tile(ranges, (m, 1)))
    return normdataSet

def classify_kNN(inX, dataSet, labels, k):
    distances = calculate_Ndistance(inX, dataSet)
    distances = distances.argsort()
    classCount = {}
    for i in range(k):
        classCount[labels[distances[i]]] = classCount.get(labels[distances[i]], 0) + 1
    sortclassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortclassCount[0][0]

def test_kNN(X_train, X_test, y_train, y_test):
    X_train, X_test = autoNorm(X_train), autoNorm(X_test)
    m = len(X_train)
    mTest = len(X_test)
    errorCount = 0
    for i in range(mTest):
        classifierResult = classify_kNN(X_test[i], X_train, y_train, 5)
        if classifierResult != y_test[i]:
            errorCount += 1
    rate = 1 - errorCount / float(mTest)
    return rate

df = pd.read_csv('iris.csv')
X = np.array(df[['sepal length', 'sepal width', 'petal length', 'petal width']])
y = classToNum(df['class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
rate = test_kNN(X_train, X_test, y_train, y_test)
print(rate)
