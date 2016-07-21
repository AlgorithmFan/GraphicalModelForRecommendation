#!usr/bin/env python
#coding:utf-8

import numpy as np
from scipy.stats import chi2


class NormalInverseWishartDistribution(object):
    def __init__(self, mu, lambda_beta, nu, psi):
        self.mu = mu
        self.lambda_beta = lambda_beta
        self.psi = psi
        self.nu = nu
        self.inv_psi = np.linalg.inv(psi)

    def sample(self):
        sigma = np.linalg.inv(self.wishartrand(self.nu, self.psi))
        return np.random.multivariate_normal(self.mu, sigma / self.lambda_beta), sigma

    def wishartrand(self, nu, phi):
        dim = phi.shape[0]
        chol = np.linalg.cholesky(phi)
        foo = np.zeros((dim, dim))

        for i in range(dim):
            for j in range(i+1):
                if i == j:
                    foo[i, j] = np.sqrt(chi2.rvs(self.nu-(i+1)+1))
                else:
                    foo[i, j] = np.random.normal(0, 1)
        return np.dot(chol, np.dot(foo, np.dot(foo.T, chol.T)))

    def posterior(self, data):
        n = len(data)
        data_mean = np.mean(data, axis=0)
        squares_sum = np.cov(data.transpose(), bias=1)
        mu_post = (self.lambda_beta * self.mu + n * data_mean) / (self.lambda_beta + n)
        beta_post = self.lambda_beta + n
        nu_post = self.nu + n
        mu0_minus_mean = self.mu - data_mean
        psi_post = self.psi + squares_sum * n + self.lambda_beta * n / (self.lambda_beta + n) * np.dot(mu0_minus_mean.transpose(), mu0_minus_mean)
        psi_post = (psi_post + np.transpose(psi_post)) / 2
        return NormalInverseWishartDistribution(mu_post, beta_post, nu_post, psi_post)

if __name__ == '__main__':
    nu = 5
    a = np.array([[1,0.5,0],[0.5,1,0],[0,0,1]])
    # x = np.array([invwishartrand(nu,a) for i in range(20000)])
