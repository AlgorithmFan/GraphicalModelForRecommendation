#!usr/bin/env python
#coding:utf-8

from scipy.sparse import dok_matrix
import numpy as np

from Recommender import Recommender

class BayesianProbabilisticMatrixFactorization(Recommender):
    '''
    Bayesian Probabilistic Matrix Factorization.
    '''
    def __init__(self, trainMatrix, testMatrix, configHandler):
        super.__init__(trainMatrix, testMatrix, configHandler)

    def initModel(self):
        self.numUsers, self.numItems = self.trainMatrix.shape()
        self.prediction = dok_matrix((self.numUsers, self.numItems))
        self.MAX_Iterations = int(self.configHandle.getParameter('BPMF', 'MAX_Iterations'))
        self.numFactors = int(self.configHandle.getParameter('BPMF', 'numFactors'))
        self.learnRate = float(self.configHandle.getParameter('BPMF', 'learning_rate'))
        self.regU = float(self.configHandle.getParameter('BPMF', 'regU'))
        self.regI = float(self.configHandle.getParameter('BPMF', 'regI'))

        self.P = np.random.normal(0, 1, size=(self.numUsers, self.numFactors))
        self.Q = np.random.normal(0, 1, size=(self.numItems, self.numFactors))


    def buildModel(self):
        pass


    def predict(self, u, i):
        pass
