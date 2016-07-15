#!usr/bin/env python
#coding:utf-8

from scipy.sparse import dok_matrix
import numpy as np

from graphicalrecommender import Recommender

class BayesianProbabilisticMatrixFactorization(Recommender):
    '''
    Bayesian Probabilistic Matrix Factorization.
    '''
    def __init__(self, train_matrix, test_matrix, config_handler):
        super.__init__(train_matrix, test_matrix, config_handler)

    def _read_config(self):
        self.max_iterations = self.config_handler.get_parameter_int('BPMF', 'max_iterations')
        self.factor_num = self.config_handler.get_parameter_int('BPMF', 'num_factors')

        self.user_normal_dist_mu0 = self.config_handler.get_parameter_float('BPMF', 'user_normal_dist_mu0')
        self.user_normal_dist_beta0 = self.config_handler.get_parameter_float('BPMF', 'user_normal_dist_beta0')

        self.item_normal_dist_mu0 = self.config_handler.get_parameter_float('BPMF', 'item_normal_dist_mu0')
        self.item_normal_dist_beta0 = self.config_handler.get_parameter_flaot('BPMF', 'item_normal_dist_beta0')

    def _init_model(self):
        self.user_num, self.item_num = self.train_matrix.shape()
        self.user_factors = np.random.normal(0, 1, size = (self.user_num, self.factor_num))
        self.item_factors = np.random.normal(0, 1, size = (self.user_num, self.factor_num))

        self.user_normal_dist_mu0 = np.zeros(self.factor_num, np.float) + self.user_normal_dist_mu0
        self.user_Wishart_dist_W0 = np.eye(np.factor_num)
        self.user_Wishart_dist_nu0 = self.factor_num

        self.item_normal_dist_mu0 = np.zeros(self.factor_num, np.float) + self.item_normal_dist_mu0
        self.item_Wishart_dist_W0 = np.eye(np.factor_num)
        self.item_Wishart_dist_nu0 = self.factor_num

    def _build_model(self):

        for i in range(self.max_iterations):
            user_factors_mu, user_factors_variance = self._sampling_hyperprameters(self.user_factors, self.user_normal_dist_mu0, self.user_Wishart_dist_nu0, self.user_Wishart_dist_W0)
            item_factors_mu, item_factors_variance = self._sampling_hyperprameters(self.item_factors, self.item_normal_dist_mu0, self.item_Wishart_dist_nu0, self.item_Wishart_dist_W0)

            for user_id in range(self.user_num):
                self.user_factors[user_id] = self._update_parameters(self.train_matrix.getrow(user_id), user_factors_mu, user_factors_variance)

            for item_id in range(self.item_num):
                self.item_factors[item_id] = self._update_parameters(self.train_matrix.getcol(item_id), item_factors_mu, item_factors_variance)

    def _sampling_hyperprameters(self, factors, normal_dist_mu0, normal_dist_beta0, Wishart_dist_nu0, Wishart_dist_W0):
        num_N = factors.shape[0]
        mean_U = np.mean(factors, axis = -1)
        variance_S = np.cov(factors, bias=1)
        mu0_minus_factors = normal_dist_mu0 - factors
        W0 = np.inv(Wishart_dist_W0) + num_N * variance_S + normal_dist_beta0 * num_N / (normal_dist_beta0 + num_N) * np.dot(mu0_minus_factors, mu0_minus_factors.transpose())
        W0_post = np.inv(W0)
        mu_post = (normal_dist_beta0 * normal_dist_mu0 + num_N * mean_U) / (normal_dist_beta0 + num_N)
        beta_post = (normal_dist_beta0 + num_N)
        nu_post = Wishart_dist_nu0 + num_N
        mu, variance_Sigma = self._sampling_normal_Wishart(mu_post, beta_post, W0_post, nu_post)
        return mu, variance_Sigma

    def _update_parameters(self, ratings, factors_mu, factors_variance):
        pass


    def gibbsSampling(mu, Alpha, userFeatrue, itemFeature):
        pass

    def _sampling_normal_Wishart(self, mu0, variance0, W0, nu0):
        pass

    def predict(self, u, i):
        return np.sum(self.user_factors[u, :] * np.item_factors[i, :])

