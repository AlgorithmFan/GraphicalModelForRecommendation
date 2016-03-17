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

        # Hyper-Parameters of Users
        mu0_U = np.zeros(self.numFactors)
        wh0_U = np.eye(self.numFactors)

        # Hyper-Parameters of Items
        mu0_V = np.zeros(self.numFactors)
        wh0_V = np.eye(self.numFactors)

        for i in range(self.MAX_Iterations):
            # Sampling from U
            mu_U, Alpha_U = self.generateHyperParameters(mu0_U, wh0_U, self.P)

            # Sampling from V
            mu_V, Alpha_V = self.generateHyperParameters(mu0_V, wh0_V, self.Q)

            for gibbs in range(2):
                self.P = self.gibbsSampling(mu_U, Alpha_U, self.P, self.Q)
                self.Q = self.gibbsSampling(mu_V, Alpha_V, self.Q, self.P)

            # calculate the loss
            loss = 0
            for u, i in self.trainMatrix.keys():
                rating = self.trainMatrix.get((u, i))
                rating_bar = self.predict(u, i)
                error = rating - rating_bar
                loss += error ** 2

        for u, i in self.testMatrix.keys():
            rate = self.testMatrix.get((u, i))
            self.prediction[u, i] = self.predict(u, i)


    def generateHyperParameters(self, mu0, wh0, FeatrueMatrix):
        '''
        FeatrueMatrix: P or Q.
        '''
        N = FeatrueMatrix.shape[0]
        beta0_asterisk = self.beta0 + self.numFactors
        nu0_asterisk = self.nu0 + N
        mean = np.mean(FeatrueMatrix, axis=0)

        S_bar = np.cov(FeatrueMatrix.transpose())
        mu0_asterisk = (self.beta0 * mu0 + N * mean) / (self.beta0 + N)
        wh0_asterisk = np.linalg.inv(wh0) + N * S_bar + self.beta0 * N / beta0_asterisk * np.outer(mean, mean)

        Alpha = self.wishart(wh0_asterisk, nu0_asterisk)

        sigma = np.linalg.cholesky(np.linalg.inv(beta0_asterisk * Alpha))
        sigma = np.linalg.inv(sigma)
        mu = np.ranom.normal(0, 1, self.numFactors)
        mu = sigma * mu + mu0_asterisk
        return mu, Alpha

    def gibbsSampling(mu, Alpha, userFeatrue, itemFeature):
        pass

    def wishart(self, scale, df):
        pass

    def predict(self, u, i):
        return np.sum(self.P[u, :] * self.Q[i, :])
