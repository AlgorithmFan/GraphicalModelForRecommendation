#!usr/bin/env python
#coding:utf-8
"""
Reference code: http://www.utstat.toronto.edu/~rsalakhu/BPMF.html
Reference paper: Salakhutdinov and Mnih, <strong>Bayesian Probabilistic Matrix Factorization using Markov Chain Monte Carlo</strong>,* ICML 2008.
"""

import numpy as np
from GraphicalRecommender import Recommender
from util.NormalWishartDistribution import NormalWishartDistribution


class BayesianProbabilisticMatrixFactorization(Recommender):
    """
    Bayesian Probabilistic Matrix Factorization.
    """
    def __init__(self, config_handler):
        Recommender.__init__(self, config_handler)

    def _read_config(self):
        self.dataset_file = self.config_handler.get_parameter_string('Dataset', 'ratings')
        self.max_iterations = self.config_handler.get_parameter_int('Parameters', 'max_iterations')
        self.factor_num = self.config_handler.get_parameter_int('Parameters', 'factor_num')

        self.user_normal_dist_mu0 = self.config_handler.get_parameter_float('Parameters', 'user_normal_dist_mu0')
        self.user_normal_dist_beta0 = self.config_handler.get_parameter_float('Parameters', 'user_normal_dist_beta0')

        self.item_normal_dist_mu0 = self.config_handler.get_parameter_float('Parameters', 'item_normal_dist_mu0')
        self.item_normal_dist_beta0 = self.config_handler.get_parameter_float('Parameters', 'item_normal_dist_beta0')

        self.rating_sigma = self.config_handler.get_parameter_float('Parameters', 'rating_sigma')

        self.logger['Result'].debug('max_iterations: {0}'.format(self.max_iterations))
        self.logger['Result'].debug('factor_num: {0}'.format(self.factor_num))
        self.logger['Result'].debug('user_normal_dist_mu0: {0}'.format(self.user_normal_dist_mu0))
        self.logger['Result'].debug('user_normal_dist_beta0: {0}'.format(self.user_normal_dist_beta0))
        self.logger['Result'].debug('item_normal_dist_mu0: {0}'.format(self.item_normal_dist_mu0))
        self.logger['Result'].debug('item_normal_dist_beta0: {0}'.format(self.item_normal_dist_beta0))

    def _init_model(self):
        self.user_num, self.item_num = self.train_matrix.shape
        self.mean_rating = np.mean(self.train_matrix.values())

        if self.config_handler.get_parameter_bool('Parameters', 'is_init_path'):
            self._load_init_model()
        else:
            self.user_factors = np.random.normal(0, 1, size=(self.user_num, self.factor_num))
            self.item_factors = np.random.normal(0, 1, size=(self.item_num, self.factor_num))

        self.user_normal_dist_mu0 = np.zeros(self.factor_num, np.float) + self.user_normal_dist_mu0
        self.user_Wishart_dist_W0 = np.eye(self.factor_num)
        self.user_Wishart_dist_nu0 = self.factor_num

        self.item_normal_dist_mu0 = np.zeros(self.factor_num, np.float) + self.item_normal_dist_mu0
        self.item_Wishart_dist_W0 = np.eye(self.factor_num)
        self.item_Wishart_dist_nu0 = self.factor_num

    def _build_model(self):
        self.previous_loss = -np.inf
        for iteration in range(self.max_iterations):
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
                    self.logger['Process'].debug('Epoch {0}: update user {1} latent factors'.format(iteration, user_id))
                    user_ratings = self.train_matrix.getrow(user_id)
                    if len(user_ratings.keys()) == 0:
                        continue
                    self.user_factors[user_id] = self._update_parameters(
                        self.item_factors, user_ratings, user_factors_mu, user_factors_variance)

                for item_id in range(self.item_num):
                    self.logger['Process'].debug('Epoch {0}: update item {1} latent factors'.format(iteration, item_id))
                    item_ratings = self.train_matrix.getcol(item_id)
                    if len(item_ratings.keys()) == 0:
                        continue
                    self.item_factors[item_id] = self._update_parameters(
                        self.user_factors, item_ratings, item_factors_mu, item_factors_variance)

                validation_rmse, test_rmse = self.__evaluate_epoch__()
                self.logger['Process'].debug('Epoch {0}: Training RMSE - {1}, Testing RMSE - {2}'.format(iteration, validation_rmse, test_rmse))
            # if self._is_converged():
            #     break

    def __evaluate_epoch__(self):
        validation_rmse = 0.0
        for user_id, item_id in self.train_matrix.keys():
            real_rating = self.train_matrix.get((user_id, item_id))
            predict_rating = self._predict(user_id, item_id)
            validation_rmse += (real_rating - predict_rating) ** 2
        self._recommend()
        results = self._evaluate()
        return np.sqrt(validation_rmse/len(self.train_matrix.keys())), results['RMSE']

    def _is_converged(self):
        loss = 0.0
        for user_id, item_id in self.train_matrix.keys():
            real_rating = self.train_matrix.get((user_id, item_id))
            predict_rating = self._predict(user_id, item_id)
            loss += (real_rating - predict_rating)**2
        loss *= 0.5
        return False

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

    def _predict(self, user_id, item_id, time_id=0):
        predict_rating = np.dot(self.user_factors[user_id, :], self.item_factors[item_id, :]) + self.mean_rating
        if predict_rating > 5:
            return 5
        elif predict_rating < 1:
            return 1
        else:
            return predict_rating

    def _run_time_sequence_algorithm(self):
        experiment_num = self.config_handler.get_parameter_int("Dataset", "experiment_num")
        time_num = self.splitter.data_model.shape[2]
        for iteration in range(experiment_num):
            self.experiment = iteration
            self.logger['Process'].debug('#'*50)
            self.logger['Process'].debug('The {0}-th experiment'.format(iteration))
            self.logger['Process'].debug('Split the dataset.')
            self.train_tensor, self.test_tensor = self.splitter.get_given_n_by_time(iteration, time_num-experiment_num)
            self.train_matrix = self.convertor.tensor_matrix(self.train_tensor)
            self.test_matrix = self.convertor.tensor_matrix(self.test_tensor)

            self.logger['Process'].debug('Initialize the model parameters.')
            self._init_model()

            self.logger['Process'].debug('Build the model.')
            self._build_model()

            self.logger['Process'].debug('Prediction.')
            self._recommend()

            self.logger['Process'].debug('Evaluation.')
            result = self._evaluate()
            for key in result:
                self.logger['Result'].debug("{0}: {1} {2}".format(iteration, key, result[key]))

    def _load_init_model(self):
        load_file = self.config_handler.get_parameter_string('Parameters', 'init_path') + 'user_factors{0}.txt'.format(self.experiment)
        self.user_factors = self._load_matrix(load_file)
        load_file = self.config_handler.get_parameter_string('Parameters', 'init_path') + 'item_factors{0}.txt'.format(self.experiment)
        self.item_factors = self._load_matrix(load_file)