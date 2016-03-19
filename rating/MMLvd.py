#!usr/bin/evn python
#coding:utf-8

from Recommender import Recommender
from scipy.sparse import dok_matrix
import numpy as np
from numpy import newaxis

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

        self.alpha = 2
        self.alpha_k = self.alpha/self.numFactors

        self.numRatings = 5

        self.theta = np.random.dirichlet(np.array([self.alpha_k for i in range(self.numFactors)]))
        self.gamma = np.zeros((self.numUsers, self.numFactors, self.numItems))

        self.sigma = np.random.normal(0, 1, size = self.numRatings)
        self.omega = np.random.normal(0, 1, size = self.numUsers)

        self.mu_vd = 1.0 / (1.0 + np.exp(-(self.omega[newaxis, ...] + self.sigma[..., newaxis])))

        self.xi = 10.0
        self.nu = 10.0
        self.phi = 2.0

    def buildModel(self):
        pass

    def EStep(self):
        gamma_nkd = np.zeros((self.numUsers, self.numFactors, self.numItems))
        beta_vkd = np.zeros((self.numRatings, self.numFactors, self.numItems))

        for u in range(self.numUsers):
            for d in range(self.numItems):
                rating = self.trainMatrix.get((u, d))
                if rating == 0:
                    gamma_nkd[u, :, d] = (beta_vkd * (1 - self.mu_vd)[..., newaxis, ...]).sum(axis=0)
                else:
                    gamma_nkd[u, :, d] = (beta_vkd[rating, :, d] * self.mu_vd[rating, d])

        qn_k = np.zeros((self.numUsers, self.numFactors))
        qn_kvd = np.zeros((self.numUsers, self.numFactors, self.numRatings, self.numItems))
        qn_vd = np.zeros((self.numUsers, self.numRatings, self.numItems))

        qn_k = np.exp(np.log(self.theta) + gamma_nkd.sum(axis=-1) - np.log(self.theta * np.exp( np.log(gamma_nkd).sum(axis=-1))))
        for u in range(self.numUsers):
            for d in range(self.numItems):
                rating = self.trainMatrix.get((u, d))
                if rating == 0:
                    qn_kvd[u, :, rating, d] = qn_k[u, :] * (1 - self.mu_vd) * beta_vkd[rating, :, d] / ((1 - self.mu_vd) * beta_vkd[:, d, :]).sum(axis=0)
                else:
                    qn_kvd[u, :, rating, d] = qn_k[u, :]

        qn_vd = qn_kvd.sum(axis=1)




    def MStep(self, qn_k, qn_kvd, qn_vd):
        self.theta = (self.alpha_k - 1 + qn_k.sum(axis=0)) / (self.numUsers - self.numFactors + self.alpha_k * self.numFactors)
        C_vdk = np.zeros((self.numRatings, self.numItems, self.numFactors))

        for u in range(self.numUsers):
            for d in range(self.numItems):
                rating = self.trainMatrix.get((u, d))
                for r in range(self.numRatings):
                    if r == rating:
                        C_vdk[r, d, :] += qn_k
                    else:
                        C_vdk[r, d, :] += qn_kvd[:, r, d]

        beta_vdk = np.zeros((self.numRatings, self.numItems, self.numFactors))
        beta_vdk = (self.phi - 1 + C_vdk) / (qn_k.sum(axis=0) - self.numRatings + self.phi * self.numRatings)

        self.sigma = self.sigma - self.learnRate * self.mu_vd * (1 - self.mu_vd) - self.sigma / self.xi
        self.omega = self.omega - self.learnRate * self.mu_vd * (1 - self.mu_vd) - self.omega / self.nu

        self.mu_vd = 1.0 / (1.0 + np.exp(-(self.omega[newaxis, ...] + self.sigma[..., newaxis])))

    def predict(self, u, i):
        return np.argmax(self.mu_vd[:, i])+1