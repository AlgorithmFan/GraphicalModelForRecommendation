#!usr/bin/evn python
#coding:utf-8

from Recommender import Recommender
from scipy.sparse import dok_matrix
import numpy as np

class MMLvd(Recommender):
    def __init__(self, trainMatrix, testMatrix, configHandler):
        super.__init__(trainMatrix, testMatrix, configHandler)


    def initModel(self):
        self.numUsers, self.numItems = self.trainMatrix.shape()
        self.prediction = dok_matrix((self.numUsers, self.numItems))
        self.MAX_Iterations = int(self.configHandler.getParameter('BPMF', 'MAX_Iterations'))
        self.numFactors = int(self.configHandler.getParameter('BPMF', 'numFactors'))

        self.beta0 = float(self.configHandler.getParameter('BPMF', 'beta0'))
        self.nu0 = float(self.configHandler.getParameter('BPMF', 'nu0'))
        self.wh0 = np.eye(self.numFactors)

        self.learnRate = float(self.configHandler.getParameter('BPMF', 'learning_rate'))
        self.regU = float(self.configHandler.getParameter('BPMF', 'regU'))
        self.regI = float(self.configHandler.getParameter('BPMF', 'regI'))

        self.P = np.random.normal(0, 1, size=(self.numUsers, self.numFactors))
        self.Q = np.random.normal(0, 1, size=(self.numItems, self.numFactors))

    def buildModel(self):
        pass

    def EStep(self):
        pass

    def MStep(self):
        pass