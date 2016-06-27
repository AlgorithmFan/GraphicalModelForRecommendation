#!usr/bin/env python
#coding:utf-8

from scipy.sparse import dok_matrix

class Recommender:
    def __init__(self, trainMatrix, testMatrix, configHandler):
        self.trainMatrix = trainMatrix
        self.testMatrix = testMatrix
        self.configHandler = configHandler

    def initModel(self):
        self.numUsers, self.numItems = self.trainMatrix.shape()
        self.prediction = dok_matrix((self.numUsers, self.numItems))
        self.MAX_Iterations = int(self.configHandle.getParameter('PMF', 'MAX_Iterations'))

    def buildModel(self):
        pass

    def predict(self):
        pass

    def evaluate(self):
        pass

    def execute(self):
        self.initModel()
        self.buildModel()
        self.evaluate()