#!usr/bin/env python
#coding:utf-8

from Recommender import Recommender

class LatentDirichletAllocation(Recommender):
    '''
    Latent Dirichlet Allocation
    '''
    def __init__(self, trainMatrix, testMatrix, configHandler):
        super.__init__(trainMatrix, testMatrix, configHandler)

    def initModel(self):
        pass

    def buildModel(self):
        pass
