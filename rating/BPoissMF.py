#!usr/bin/env python
#coding:utf-8

'''
Paper: Prem Gopalan, et al. Scalable Recommendation with Poisson Factorization.
Github: https://github.com/mertterzihan/PMF/blob/master/Code/PoissonFactorization.py#L47
http://www.hongliangjie.com/2015/08/17/poisson-matrix-factorization/

Author: Haidong Zhang
Date: April 24, 2016
'''

from Recommender import Recommender
import numpy as np
from scipy.sparse import dok_matrix
from util import normalize
from random import shuffle
from itertools import product
from scipy.stats import poisson
from scipy.stats import gamma as gammafun
import sys

def gammaRnd(shape, scale, size=None):
    result = np.random.gamma(shape, scale, size)
    return result

def poissonRnd(scale, size=None):
    result = np.random.poisson(scale, size)
    return result

class BPoissMF(Recommender):
    def __init__(self, trainMatrix, testMatrix, configHandler):
        Recommender.__init__(trainMatrix, testMatrix, configHandler)

    def initModel(self):
        self.numUsers, self.numItems = self.trainMatrix.shape()
        self.prediction = dok_matrix((self.numUsers, self.numItems))
        self.MAX_Iterations = int(self.configHandler.getParameter('BPoissMF', 'MAX_Iterations'))
        self.numFactors = int(self.configHandler.getParameter('BPoissMF', 'numFactors'))
        self.threshold = float(self.configHandler.getParameter('BPoissMF', 'threshold'))

        # Get the Parameters
        self.a = float(self.configHandler.getParameter('BPoissMF', 'a'))
        self.ap = float(self.configHandler.getParameter('BPoissMF', 'ap'))
        self.bp = float(self.configHandler.getParameter('BPoissMF', 'bp'))

        self.c = float(self.configHandler.getParameter('BPoissMF', 'c'))
        self.cp = float(self.configHandler.getParameter('BPoissMF', 'cp'))
        self.dp = float(self.configHandler.getParameter('BPoissMF', 'dp'))

        # Init xi
        self.xi = gammaRnd(self.ap, self.ap/self.bp, size=self.numUsers)
        # Init theta
        self.theta = np.zeros((self.numUsers, self.numFactors))
        for i in range(self.numUsers):
            self.theta[i, :] = gammaRnd(self.a, self.xi[i])

        # Init eta
        self.eta = gammaRnd(self.cp, self.cp/self.dp, size=self.numItems)
        #Init beta
        self.beta = np.zeros((self.numItems, self.numFactors))
        for i in range(self.numItems):
            self.beta[i, :] = gammaRnd(self.c, self.eta[i])

        # Init z
        self.zs = np.zeros((self.numUsers, self.numItems, self.numFactors))
        for user_id, item_id in self.trainMatrix.keys():
            p = self.theta[user_id, :] * self.beta[item_id, :]
            p /= np.sum(p)
            self.zs[user_id, item_id, :] = np.random.multinomial(self.trainMatrix[user_id, item_id], p)

    def sample(self, ):
        ''''''
        self.loglikelihood = []
        for curr_iter in xrange(self.MAX_Iterations):
            'Gibbs Sampling.'
            randUsers = range(self.numUsers)
            randTopics = range(self.numFactors)
            randItems = range(self.numItems)

            # Sample theta
            shuffle(randUsers)
            for user_id in randUsers:
                shuffle(randTopics)
                for topic_id in randTopics:
                    self.theta[user_id, topic_id] = gammaRnd(self.a + np.sum(self.zs[user_id, :, topic_id]),
                                                             self.xi[user_id] + np.sum(self.beta[:, topic_id]))

            # Sample beta
            shuffle(randItems)
            for item_id in randItems:
                shuffle(randTopics)
                for topic_id in randTopics:
                    self.beta[item_id, topic_id] = gammaRnd(self.c + np.sum(self.zs[:, item_id, topic_id]),
                                                            self.eta[item_id] + np.sum(self.theta[:, topic_id]))

            # Sample xi
            shuffle(randUsers)
            for user_id in randUsers:
                self.xi[user_id] = gammaRnd(self.ap + self.numFactors*self.a, self.b + self.theta[user_id, :].sum())

            # Sample eta
            shuffle(randItems)
            for item_id in randItems:
                self.eta[item_id] = gammaRnd(self.cp + self.numFactors*self.c, self.d + self.beta[item_id, :].sum())

            # Sample zs
            nonzeros = self.trainMatrix.keys()
            randNonZeros = shuffle(len(nonzeros))
            for pair_id in randNonZeros:
                user_id, item_id = nonzeros[pair_id]
                p = self.theta[user_id, :] * self.beta[item_id, :]
                p /= p.sum()
                self.zs[user_id, item_id, :] = np.random.multinomial(self.trainMatrix[user_id, item_id], p)










if __name__ == '__main__':
    bnprec = BPoissMF()