#!usr/bin/env python
#coding:utf-8

import numpy as np
from scipy.sparse import dok_matrix
from util import readConfig
from Recommender import Recommender

class ProbabilisticMatrixFactorization(Recommender):
    def __init__(self, trainMatrix, testMatrix, configHandle):
        super.__init__(trainMatrix, testMatrix, configHandle)

    def initModel(self):
        self.numUsers, self.numItems = self.trainMatrix.shape()
        self.prediction = dok_matrix((self.numUsers, self.numItems))
        self.MAX_Iterations = int(self.configHandle.getParameter('PMF', 'MAX_Iterations'))
        self.numFactors = int(self.configHandle.getParameter('PMF', 'numFactors'))
        self.learnRate = float(self.configHandle.getParameter('PMF', 'learning_rate'))
        self.regU = float(self.configHandle.getParameter('PMF', 'regU'))
        self.regI = float(self.configHandle.getParameter('PMF', 'regI'))

        self.P = np.random.normal(0, 1, size=(self.numUsers, self.numFactors))
        self.Q = np.random.normal(0, 1, size=(self.numItems, self.numFactors))


    def buildModel(self):
        trainMatrix = dok_matrix((4,3))
        trainMatrix.get()

        oldLoss = np.inf
        for iteration in range(self.MAX_Iterations):
            loss = 0.0

            for u, i in self.trainMatrix.keys():
                rate = self.trainMatrix.get((u, i))
                rate_post = self.predict(u, i)
                error = rate - rate_post
                loss += error ** 2

                self.P[u, :] += self.learnRate * (error * self.Q[i, :] - self.regU * self.P[u, :])
                self.Q[i, :] += self.learnRate * (error * self.P[u, :] - self.regI * self.Q[i, :])

                loss += self.regU * np.sum(self.P[u, :] ** 2) + self.regI * np.sum(self.Q[i, :] ** 2)

            if np.abs(oldLoss - loss) < 0.001:
                break
            else:
                oldLoss = loss

        for u, i in self.testMatrix.keys():
            rate = self.testMatrix.get((u, i))
            self.prediction[u, i] = self.predict(u, i)

    def predict(self, u, i):
        return np.sum(self.P[u, :] * self.Q[i, :])


