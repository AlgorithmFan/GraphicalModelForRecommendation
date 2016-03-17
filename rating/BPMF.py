#!usr/bin/env python
#coding:utf-8

from Recommender import Recommender

class BayesianProbabilisticMatrixFactorization(Recommender):
    '''
    Bayesian Probabilistic Matrix Factorization.
    '''
    def __init__(self, trainMatrix, testMatrix, configHandler):
        super.__init__(trainMatrix, testMatrix, configHandler)

    def initModel(self):
        pass


    def buildModel(self):
        pass
