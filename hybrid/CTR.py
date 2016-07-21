#!usr/bin/env python
#coding:utf-8

'''
Collaborative Topic Regression
'''

from Recommender import Recommender
from scipy.sparse import dok_matrix
import numpy as np
from util import normalize


class CTR(Recommender):
    def __init__(self, train_matrix, test_matrix, config_handler):
        super.__init__(train_matrix, test_matrix, config_handler)



    def initModel(self):
        ''''''
        self.numUsers, self.numItems = self.trainMatrix.shape()
        self.prediction = dok_matrix((self.numUsers, self.numItems))
        self.MAX_Iterations = int(self.configHandler.getParameter('CTR', 'MAX_Iterations'))
        self.numFactors = int(self.configHandler.getParameter('CTR', 'numFactors'))
        self.threshold = float(self.configHandler.getParameter('CTR', 'threshold'))

        self.U = np.zeros((self.numUsers, self.numFactors))
        self.V = np.zeros((self.numItems, self.numFactors))

    def buildModel(self, corpus):
        '''
        corpus: document * words.
        '''

        # Update U



        # Update V




        # Update theta


    def predict(self):
        ''''''
        