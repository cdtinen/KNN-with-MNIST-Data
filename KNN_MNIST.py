"""
Program: KNN_MNIST.py
Author: Collin Tinen
Description: Reads data from the MNIST dataset and uses the KNN algorithm to predict the labels of the test set.
"""

import pandas as pd
import math

training_data = pd.read_csv('MNIST_train.csv')
test_data = pd.read_csv('MNIST_test.csv')

length = len(training_data)
train = []
for row in range(length):
    train.append(training_data.loc[row])
global accuracy
accuracy = 0


def EuclDistance(point1, point2, length):
    # calculates Euclidean Distance
    distance = 0
    for i in range(length - 1):
        distance += pow((point1[i] - point2[i]), 2)
    EuclideanDistance = math.sqrt(distance)
    return EuclideanDistance


def getSortedDistance(testItem, length):
    # puts all distances into a list, sorted in descending order
    distances = []
    for i in range(length - 1):
        dist = EuclDistance(train[i][1:], testItem[1:], len(training_data.columns))
        distances.append((dist, i))
    return sorted(distances)


def KNN(testItem, k, sorteddist):
    neighborindex = []
    neighborlabels = []
    neighbors = []
    global accuracy

    for i in range(k):
        neighbors.append(sorteddist[i])
        neighborindex.append(neighbors[i][1])
    for j in range(len(neighborindex)):
        neighborlabels.append(train[neighborindex[j]][0])

    # weigh the votes
    weights = []
    for dist in range(len(neighbors)):
        weights.append(((1 / pow(sorteddist[dist][0], 2)), neighborlabels[dist]))

    # find the label with the highest vote weight
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    labelvotes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for vote in weights:
        temp = list(vote)
        for z in range(len(labels)):
            if temp[1] == labels[z]:
                labelvotes[z] += temp[0]
    highestvote = max(labelvotes)
    # use the index of the highest vote to match it with the index of the label to get the predicted label
    predictedlabel = labels[labelvotes.index(highestvote)]

    # test accuracy
    if testItem[0] == predictedlabel:
        print("Desired Class: {} Computed Class: {}".format(testItem[0], predictedlabel))
        accuracy += 1
    else:
        print("Desired Class: {} Computed Class: {}".format(testItem[0], predictedlabel))
    if testItem.equals(test_data.loc[len(test_data) - 1]):
        print("Accuracy Rate: {}%".format((accuracy / len(test_data)) * 100))
        print("Number of misclassified test samples: {}".format(len(test_data) - accuracy))
        print("Total number of test samples: {}".format(len(test_data)))


if __name__ == '__main__':
    k = 7
    print("K =", k)
    for x in range(len(test_data)):
        KNN(test_data.loc[x], k, getSortedDistance(test_data.loc[x], length))
