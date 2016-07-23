#!usr/bin/env python
#coding:utf-8
"""
Reference code: http://www.utstat.toronto.edu/~rsalakhu/BPMF.html
Reference paper: Salakhutdinov and Mnih, <strong>Bayesian Probabilistic Matrix Factorization using Markov Chain Monte Carlo</strong>,* ICML 2008.
"""

import numpy as np
import codecs
from GraphicalRecommender import Recommender
from util.NormalWishartDistribution import NormalWishartDistribution
from scipy.sparse import dok_matrix


class BayesianProbabilisticMatrixFactorization(Recommender):
    """
    Bayesian Probabilistic Matrix Factorization.
    """
    def __init__(self, recommender_context):
        Recommender.__init__(self, recommender_context)

    def _read_cfg(self):
        self.user_normal_dist_mu0_init = self.config_handler['Parameters', 'user_normal_dist_mu0', 'float']
        self.user_normal_dist_beta0_init = self.config_handler['Parameters', 'user_normal_dist_beta0', 'float']
        self.user_Wishart_dist_W0_init = self.config_handler['Parameters', 'user_Wishart_dist_W0', 'float']

        self.item_normal_dist_mu0_init = self.config_handler['Parameters', 'item_normal_dist_mu0', 'float']
        self.item_normal_dist_beta0_init = self.config_handler['Parameters', 'item_normal_dist_beta0', 'float']
        self.item_Wishart_dist_W0_init = self.config_handler['Parameters', 'item_Wishart_dist_W0', 'float']

        self.rating_sigma_init = self.config_handler['Parameters', 'rating_sigma', 'float']

    def _init_model(self):
        self.user_num, self.item_num = self.train_matrix.shape
        self.mean_rating = np.mean(self.train_matrix.values())

        self.predictions = dok_matrix((self.user_num, self.item_num))

        if self.config_handler['Output', 'is_load', 'bool']:
            self._load_model()
            assert(self.user_factors.shape[1] == self.item_factors.shape[1])
            self.factor_num = self.user_factors.shape[1]
        else:
            self._read_cfg()

        if self.config_handler['Parameters', 'is_init_path', 'bool']:
            self._load_init_model()
        else:
            self.factor_num = self.config_handler['Parameters', 'factor_num', 'int']
            self.user_factors = np.random.normal(0, 1, size=(self.user_num, self.factor_num))
            self.item_factors = np.random.normal(0, 1, size=(self.item_num, self.factor_num))

        self.markov_num = 0
        validation_rmse, test_rmse = self.__evaluate_epoch__()
        self.logger['Process'].debug('Epoch {0}: Training RMSE - {1}, Testing RMSE - {2}'.format(0, validation_rmse, test_rmse))

        self.user_normal_dist_mu0 = np.zeros(self.factor_num, np.float) + self.user_normal_dist_mu0_init
        self.user_normal_dist_beta0 = self.user_normal_dist_beta0_init
        self.user_Wishart_dist_W0 = np.eye(self.factor_num) * self.user_Wishart_dist_W0_init
        self.user_Wishart_dist_nu0 = self.factor_num

        self.item_normal_dist_mu0 = np.zeros(self.factor_num, np.float) + self.item_normal_dist_mu0_init
        self.item_normal_dist_beta0 = self.item_normal_dist_beta0_init
        self.item_Wishart_dist_W0 = np.eye(self.factor_num) * self.item_Wishart_dist_W0_init
        self.item_Wishart_dist_nu0 = self.factor_num

        self.rating_sigma = self.rating_sigma_init

    def _build_model(self):
        user_train_matrix = dict()
        item_train_matrix = dict()
        for user_id, item_id in self.train_matrix.keys():
            user_train_matrix.setdefault(user_id, dok_matrix((1, self.item_num)))
            user_train_matrix[user_id][0, item_id] = self.train_matrix.get((user_id, item_id))
            item_train_matrix.setdefault(item_id, dok_matrix((1, self.user_num)))
            item_train_matrix[item_id][0, user_id] = self.train_matrix.get((user_id, item_id))

        self.previous_loss = -np.inf
        max_iterations = self.config_handler['Parameters', 'max_iterations', 'int']
        for iteration in range(max_iterations):
            self.logger['Process'].debug('Epoch {0}: update hyper-parameters'.format(iteration))
            user_factors_mu, user_factors_variance = \
                self._sampling_hyperparameters(self.user_factors, self.user_normal_dist_mu0, self.user_normal_dist_beta0,
                                               self.user_Wishart_dist_nu0, self.user_Wishart_dist_W0)
            item_factors_mu, item_factors_variance = \
                self._sampling_hyperparameters(self.item_factors, self.item_normal_dist_mu0, self.item_normal_dist_beta0,
                                              self.item_Wishart_dist_nu0, self.item_Wishart_dist_W0)

            self.logger['Process'].debug('Epoch {0}: update latent factors'.format(iteration))
            for gibbs_iteration in range(2):
                for user_id in range(self.user_num):
                    user_ratings = user_train_matrix[user_id] if user_id in user_train_matrix else dict()
                    if len(user_ratings.keys()) == 0:
                        continue
                    self.user_factors[user_id] = self._update_parameters(
                        self.item_factors, user_ratings, user_factors_mu, user_factors_variance)

                for item_id in range(self.item_num):
                    item_ratings = item_train_matrix[item_id] if item_id in item_train_matrix else dict()
                    if len(item_ratings.keys()) == 0:
                        continue
                    self.item_factors[item_id] = self._update_parameters(
                        self.user_factors, item_ratings, item_factors_mu, item_factors_variance)

                validation_rmse, test_rmse = self.__evaluate_epoch__()
                self.logger['Process'].debug('Epoch {0}: Training RMSE - {1}, Testing RMSE - {2}'.format(iteration, validation_rmse, test_rmse))

    def __evaluate_epoch__(self):
        validation_rmse = 0.0
        for user_id, item_id in self.train_matrix.keys():
            real_rating = self.train_matrix.get((user_id, item_id))
            predict_rating = self._predict(user_id, item_id)
            validation_rmse += (real_rating - predict_rating) ** 2
        self._recommend()
        results = self._evaluate()
        return np.sqrt(validation_rmse/len(self.train_matrix.keys())), results['RMSE']

    def _sampling_hyperparameters(self, factors, normal_dist_mu0, normal_dist_beta0, Wishart_dist_nu0, Wishart_dist_W0):
        num_N = factors.shape[0]
        mean_U = np.mean(factors, axis=0)
        variance_S = np.cov(factors.transpose(), bias=1)
        mu0_minus_factors = normal_dist_mu0 - mean_U
        mu0_minus_factors = np.reshape(mu0_minus_factors, (mu0_minus_factors.shape[0], 1))

        W0 = np.linalg.inv(Wishart_dist_W0) + num_N * variance_S \
             + normal_dist_beta0 * num_N / (normal_dist_beta0 + num_N) * np.dot(mu0_minus_factors, mu0_minus_factors.transpose())
        W0_post = np.linalg.inv(W0)
        W0_post = (W0_post + W0_post.transpose()) / 2

        mu_post = (normal_dist_beta0 * normal_dist_mu0 + num_N * mean_U) / (normal_dist_beta0 + num_N)
        beta_post = (normal_dist_beta0 + num_N)
        nu_post = Wishart_dist_nu0 + num_N
        normal_Wishart_distribution = NormalWishartDistribution(mu_post, beta_post, nu_post, W0_post)
        mu, sigma = normal_Wishart_distribution.sample()
        return mu, sigma

    def _update_parameters(self, factors, ratings, factors_mu, factors_variance):
        index = np.array([col_id for row_id, col_id in ratings.keys()])
        VVT = np.dot(factors[index, :].transpose(), factors[index, :])
        sigma = factors_variance + self.rating_sigma * VVT
        sigma_inv = np.linalg.inv(sigma)

        rating_values = np.array(ratings.values()) - self.mean_rating
        VR = np.dot(factors[index, :].transpose(), np.reshape(rating_values, newshape=(rating_values.shape[0], 1)))
        mu_right = self.rating_sigma * VR + np.dot(factors_variance, np.reshape(factors_mu, newshape=(factors_mu.shape[0], 1)))
        mu = np.dot(sigma_inv, mu_right)
        mu = np.reshape(mu, newshape=(mu.shape[0], ))
        return np.random.multivariate_normal(mu, sigma_inv)

    def _recommend(self):
        for user_id, item_id in self.test_matrix.keys():
            predict_rating = self._predict(user_id, item_id) + self.predictions[user_id, item_id] * self.markov_num
            self.predictions[user_id, item_id] = predict_rating / (self.markov_num + 1)
        self.markov_num += 1

    def _predict(self, user_id, item_id, time_id=0):
        predict_rating = np.dot(self.user_factors[user_id, :], self.item_factors[item_id, :]) + self.mean_rating
        if predict_rating > 5:
            return 5
        elif predict_rating < 1:
            return 1
        else:
            return predict_rating

    def _load_init_model(self):
        load_path = self.config_handler["Output", "load_path", "string"]
        load_file = load_path + "PMF_{0}.txt".format(self.recommender_context.experiment_id)

        with codecs.open(load_file, mode='r', encoding='utf-8') as read_fp:
            for line in read_fp:
                if line.startswith('factor_num'):
                    self.factor_num = int(line.split(':')[1].strip())
                elif line.startswith('user_factor'):
                    self.user_factors = self._load_matrix(read_fp)
                elif line.startswith('item_factor'):
                    self.item_factors = self._load_matrix(read_fp)

    def _save_result(self, result):
        self.logger['Result'].debug('factor_num: {0}'.format(self.factor_num))

        self.logger['Result'].debug('user_normal_dist_mu0: {0}'.format(self.user_normal_dist_mu0_init))
        self.logger['Result'].debug('user_normal_dist_beta0: {0}'.format(self.user_normal_dist_beta0_init))
        self.logger['Result'].debug('user_Wishart_dist_W0: {0}'.format(self.user_Wishart_dist_W0_init))

        self.logger['Result'].debug('item_normal_dist_mu0: {0}'.format(self.item_normal_dist_mu0_init))
        self.logger['Result'].debug('item_normal_dist_beta0: {0}'.format(self.item_normal_dist_beta0_init))
        self.logger['Result'].debug('item_Wishart_dist_W0: {0}'.format(self.item_Wishart_dist_W0_init))

        Recommender._save_result(self, result)

    def _save_model(self):
        save_path = self.config_handler["Output", "save_path", "string"]
        save_file = save_path + "BPMF_{0}.txt".format(self.recommender_context.experiment_id)

        with codecs.open(save_file, mode='w', encoding='utf-8') as write_fp:
            write_fp.write('factor_num: {0}\n'.format(self.factor_num))
            write_fp.write('user_normal_dist_mu0: {0}\n'.format(self.user_normal_dist_mu0_init))
            write_fp.write('user_normal_dist_beta0: {0}\n'.format(self.user_normal_dist_beta0_init))
            write_fp.write('user_Wishart_dist_W0: {0}\n'.format(self.user_Wishart_dist_W0_init))
            write_fp.write('item_normal_dist_mu0: {0}\n'.format(self.item_normal_dist_mu0_init))
            write_fp.write('item_normal_dist_beta0: {0}\n'.format(self.item_normal_dist_beta0_init))
            write_fp.write('item_Wishart_dist_W0: {0}\n'.format(self.item_Wishart_dist_W0_init))

            write_fp.write('user_factors \n')
            self._save_matrix(self.user_factors, write_fp)

            write_fp.write('item_factors \n')
            self._save_matrix(self.item_factors, write_fp)

    def _load_model(self):
        load_path = self.config_handler["Output", "load_path", "string"]
        load_file = load_path + "PMF_{0}.txt".format(self.recommender_context.experiment_id)

        with codecs.open(load_file, mode='r', encoding='utf-8') as read_fp:
            for line in read_fp:
                if line.startswith('factor_num'):
                    self.factor_num = int(line.split(':')[1].strip())
                elif line.startswith('user_normal_dist_mu0'):
                    self.user_normal_dist_mu0_init = float(line.split(':')[1].strip())
                elif line.startswith('user_normal_dist_beta0'):
                    self.user_normal_dist_beta0_init = float(line.split(':')[1].strip())
                elif line.startswith('user_Wishart_dist_W0'):
                    self.user_Wishart_dist_W0_init = float(line.split(':')[1].strip())
                elif line.startswith('item_normal_dist_mu0'):
                    self.item_normal_dist_mu0_init = float(line.split(':')[1].strip())
                elif line.startswith('item_normal_dist_beta0'):
                    self.item_normal_dist_beta0_init = float(line.split(':')[1].strip())
                elif line.startswith('item_Wishart_dist_W0'):
                    self.item_Wishart_dist_W0_init = float(line.split(':')[1].strip())
                elif line.startswith('user_factor'):
                    self.user_factors = self._load_matrix(read_fp)
                elif line.startswith('item_factor'):
                    self.item_factors = self._load_matrix(read_fp)