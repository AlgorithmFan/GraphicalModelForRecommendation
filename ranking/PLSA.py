#!usr/bin/env python
#coding:utf-8

from Recommender import Recommender
import numpy as np
from scipy.sparse import dok_matrix
from util import normalize

class PLSA(Recommender):
    def __init__(self, trainMatrix, testMatrix, configHandler):
        super.__init__(trainMatrix, testMatrix, configHandler)

    def initModel(self):
        self.numUsers, self.numItems = self.trainMatrix.shape()
        self.prediction = dok_matrix((self.numUsers, self.numItems))
        self.MAX_Iterations = int(self.configHandler.getParameter('PLSA', 'MAX_Iterations'))
        self.numFactors = int(self.configHandler.getParameter('PLSA', 'numFactors'))

        self.X = np.random.uniform(0, 1, size=(self.numUsers, self.numFactors))      #  P(z|x)
        self.X = normalize(self.X)

        self.Y = np.random.uniform(0, 1, size=(self.numItems, self.numFactors))      #  P(y|z)
        self.Y = normalize(self.Y)

        self.Q = np.zeros((self.numUsers, self.numFactors, self.numItems))   # P(y,z|x)

    def buildModel(self):
        ''''''
        self.initModel()
        oldLikelihood = np.inf
        for iteration in range(self.MAX_Iterations):
            ''''''
            print 'Iteration {}'.format(iteration)
            self.eStep()    # E-Step
            self.mStep()    # M-Step
            likelihood = self.likelihood()

            if likelihood - oldLikelihood < self.threshold:
                break
            else:
                oldLikelihood = likelihood


    def eStep(self):
        ''''''
        self.Q = self.X[..., np.newaxis, ...] * self.Y[np.newaxis, ...]
        self.Q = self.Q / np.sum(self.Q, axis=-1)[..., np.newaxis]


    def mStep(self):
        ''''''
        probability = self.Q * self.trainMatrix[..., np.newaxis]
        self.X = np.sum(probability, axis=1) / np.sum(np.sum(probability, axis=0), axis=0)[np.newaxis, ...]
        self.Y = np.sum(probability, axis=0) / np.sum(np.sum(probability, axis=0), axis=0)[np.newaxis, ...]

    def likelihood(self):
        ''''''
        result = 0.00
        logX = np.log(self.X)
        logY = np.log(self.Y)
        for user_id, item_id in self.trainMatrix.keys():
            result += np.log(self.Q[user_id, item_id, :] * (logX[user_id, :] + logY[item_id, :]))
        return  result / len(self.trainMatrix.keys())

    def RegularizedLikelihood(self):
        ''''''
        result = 0.00
        logX = np.log(self.X)
        logY = np.log(self.Y)
        logQ = np.log(self.Q)
        for user_id, item_id in self.trainMatrix.keys():
            result += np.sum(self.Q[user_id, item_id, :] * (logX[user_id, :] + logY[item_id, :]))
            result += np.sum(self.Q[user_id, item_id, :] * logQ[user_id, item_id, :])
        return  result / len(self.trainMatrix.keys())


    def ranking(self, user_id, item_id):
        ''''''
        return np.sum(self.X[user_id, :] * self.Y[item_id, :] * self.Z)
