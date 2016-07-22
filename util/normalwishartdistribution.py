#!usr/bin/env python
#coding:utf-8

import numpy as np
import random
from scipy.stats import chi2


class NormalWishartDistribution(object):
    def __init__(self, mu, lambda_beta, nu, psi):
        self.mu = mu
        self.lambda_beta = lambda_beta
        self.psi = psi
        self.nu = nu
        # self.inv_psi = np.linalg.inv(psi)

    def sample(self):
        sigma = np.linalg.inv(self.wishartrand(self.nu, self.psi))
        mu = np.random.multivariate_normal(self.mu, sigma / self.lambda_beta)
        return mu, sigma

    def wishartrand(self, nu, sigma, C=None):
        """Return a sample from a Wishart distribution."""
        if C == None:
            C = np.linalg.cholesky(sigma)
        D = sigma.shape[0]
        a = np.zeros((D, D), dtype=np.float32)
        for r in xrange(D):
            if r != 0:
                a[r, :r] = np.random.normal(size=(r,))
            a[r, r] = np.sqrt(random.gammavariate(0.5*(nu - D + 1), 2.0))
        return np.dot(np.dot(np.dot(C, a), a.T), C.T)

    def wishartrand1(self, nu, phi):
        dim = phi.shape[0]
        chol = np.linalg.cholesky(phi)
        foo = np.zeros((dim, dim))

        for i in range(dim):
            for j in range(i+1):
                if i == j:
                    foo[i, j] = np.sqrt(chi2.rvs(nu-(i+1)+1))
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
        psi_post = np.linalg.inv(self.psi) + squares_sum * n + self.lambda_beta * n / (self.lambda_beta + n) * np.dot(mu0_minus_mean.transpose(), mu0_minus_mean)
        psi_post = np.linalg.inv(psi_post)
        psi_post = (psi_post + np.transpose(psi_post)) / 2
        return NormalWishartDistribution(mu_post, beta_post, nu_post, psi_post)

if __name__ == '__main__':
    nu = 5
    sigma = np.array([[1, 0.5], [0.5, 2]])
    df = 10
    np.random.seed(1)
    nwd = NormalWishartDistribution(0, 0, df, sigma)
    sigma1 = nwd.wishartrand(df, sigma)
    print sigma1
    print np.linalg.inv(sigma1)
