#!usr/bin/env python
#coding:utf-8

"""
Reference Paper: Liang Xiong et al.
    <strong>Temporal Collaborative Filtering with Bayesian Probabilistic Tensor Factorization</strong>,
Reference Code: https://www.cs.cmu.edu/~lxiong/bptf/bptf.html
"""

import numpy as np
import codecs
from scipy.sparse import dok_matrix
from GraphicalRecommender import Recommender
from util.NormalWishartDistribution import NormalWishartDistribution


class BayesianProbabilisticTensorFactorization(Recommender):
    def __init__(self, config_handler):
        Recommender.__init__(self, config_handler)

    def _read_cfg(self):

        self.user_normal_dist_mu0_init = self.config_handler['Parameters', 'user_normal_dist_mu0', 'float']
        self.user_normal_dist_beta0_init = self.config_handler['Parameters', 'user_normal_dist_beta0', 'float']
        self.user_Wishart_dist_W0_init = self.config_handler['Parameters', 'user_Wishart_dist_W0', 'float']

        self.item_normal_dist_mu0_init = self.config_handler['Parameters', 'item_normal_dist_mu0', 'float']
        self.item_normal_dist_beta0_init = self.config_handler['Parameters', 'item_normal_dist_beta0', 'float']
        self.item_Wishart_dist_W0_init = self.config_handler['Parameters', 'item_Wishart_dist_W0', 'float']

        self.time_normal_dist_mu0_init = self.config_handler['Parameters', 'time_normal_dist_mu0', 'float']
        self.time_normal_dist_beta0_init = self.config_handler['Parameters', 'time_normal_dist_beta0', 'float']
        self.time_Wishart_dist_W0_init = self.config_handler['Parameters', 'time_Wishart_dist_W0', 'float']

        self.rating_sigma_init = self.config_handler['Parameters', 'rating_sigma', 'float']

    def _init_model(self):
        self.user_num, self.item_num, self.time_num = self.train_tensor.shape()
        self.mean_rating = np.mean(self.train_tensor.values())

        self.predictions = dok_matrix((self.user_num, self.item_num, self.time_num))

        if self.config_handler['Parameters', 'is_load', 'bool']:
            self._load_model()
            assert(self.user_factors.shape[1] == self.item_factors.shape[1] and self.item_factors.shape[1] == self.time_factors.shape[1])
            self.factor_num = self.user_factors.shape[1]
        else:
            self._read_cfg()

        # initialize the latent factors of user, item and time.
        if self.config_handler['Parameters', 'is_init_path', 'bool']:
            self._load_init_model()
        else:
            self.factor_num = self.config_handler['Parameters', 'factor_num', 'int']
            self.user_factors = np.random.normal(0, 1, size=(self.user_num, self.factor_num))
            self.item_factors = np.random.normal(0, 1, size=(self.item_num, self.factor_num))
        self.time_factors = np.random.normal(0, 1, size=(self.time_num, self.factor_num))

        self.markov_num = 0
        validation_rmse, test_rmse = self.__evaluate_epoch__()
        self.logger['Process'].debug('Epoch {0}: Training RMSE - {1}, Testing RMSE - {2}'.format(0, validation_rmse, test_rmse))

        # get the user parameters
        self.user_normal_dist_mu0 = np.zeros(self.factor_num, np.float) + self.user_normal_dist_mu0_init
        self.user_normal_dist_beta0 = self.user_normal_dist_beta0_init
        self.user_Wishart_dist_W0 = np.eye(self.factor_num) * self.user_Wishart_dist_W0_init
        self.user_Wishart_dist_nu0 = self.factor_num

        # get the item parameters
        self.item_normal_dist_mu0 = np.zeros(self.factor_num, np.float) + self.item_normal_dist_mu0_init
        self.item_normal_dist_beta0 = self.item_normal_dist_beta0_init
        self.item_Wishart_dist_W0 = np.eye(self.factor_num) * self.item_Wishart_dist_W0_init
        self.item_Wishart_dist_nu0 = self.factor_num

        # get the time parameters
        self.time_normal_dist_mu0 = np.zeros(self.factor_num, np.float) + self.time_normal_dist_mu0_init
        self.time_normal_dist_beta0 = self.time_normal_dist_beta0_init
        self.time_Wishart_dist_W0 = np.eye(self.factor_num) * self.time_Wishart_dist_W0_init
        self.time_Wishart_dist_nu0 = self.factor_num

        self.rating_sigma = self.rating_sigma_init

    def _build_model(self):

        # Speed up the process of gibbs sampling
        train_matrix_by_user, train_matrix_by_item, train_matrix_by_time = dict(), dict(), dict()
        for user_id, item_id, time_id in self.train_tensor.keys():
            train_matrix_by_user.setdefault(user_id, dok_matrix((self.item_num, self.time_num)))
            train_matrix_by_user[user_id][item_id, time_id] = self.train_tensor[user_id, item_id, time_id]

            train_matrix_by_item.setdefault(item_id, dok_matrix((self.user_num, self.time_num)))
            train_matrix_by_item[item_id][user_id, time_id] = self.train_tensor[user_id, item_id, time_id]

            train_matrix_by_time.setdefault(time_id, dok_matrix((self.user_num, self.item_num)))
            train_matrix_by_time[time_id][user_id, item_id] = self.train_tensor[user_id, item_id, time_id]

        max_iterations = self.config_handler['Parameters', 'max_iterations', 'int']
        for iteration in range(max_iterations):
            user_factors_mu, user_factors_variance = \
                self._sampling_hyperparameters(self.user_factors, self.user_normal_dist_mu0, self.user_normal_dist_beta0,
                                               self.user_Wishart_dist_nu0, self.user_Wishart_dist_W0)
            item_factors_mu, item_factors_variance = \
                self._sampling_hyperparameters(self.item_factors, self.item_normal_dist_mu0, self.item_normal_dist_beta0,
                                               self.item_Wishart_dist_nu0, self.item_Wishart_dist_W0)

            time_factors_mu, time_factors_variance = \
                self._sampling_time_hyperparameters(self.time_factors, self.time_normal_dist_mu0, self.time_normal_dist_beta0,
                                                    self.time_Wishart_dist_nu0, self.time_Wishart_dist_W0)

            for gibbs_iteration in range(2):
                for user_id in range(self.user_num):
                    item_time_matrix = train_matrix_by_user[user_id]
                    if len(item_time_matrix.keys()) < 1:
                        continue
                    self.user_factors[user_id] = self._update_parameters(
                        self.item_factors, self.time_factors, item_time_matrix, user_factors_mu, user_factors_variance)

                for item_id in range(self.item_num):
                    user_time_matrix = train_matrix_by_item[item_id]
                    if len(user_time_matrix.keys()) < 1:
                        continue
                    self.item_factors[item_id] = self._update_parameters(
                        self.user_factors, self.time_factors, user_time_matrix, item_factors_mu, item_factors_variance)

                for time_id in range(self.time_num):
                    user_item_matrix = train_matrix_by_time[time_id]
                    if len(user_item_matrix.keys()) < 1:
                        continue
                    self.time_factors[time_id] = self._update_time_parameters(
                        self.user_factors, self.item_factors, self.time_factors, user_item_matrix, time_factors_mu, time_factors_variance, time_id)

                validation_rmse, test_rmse = self.__evaluate_epoch__()
                self.logger['Process'].debug('Epoch {0}: Training RMSE - {1}, Testing RMSE - {2}'.format(iteration, validation_rmse, test_rmse))

    def run(self):
        self.logger['Process'].debug('Get the train dataset')
        self.train_tensor = self.recommender_context.get_data_model().get_data_splitter().get_train_data()
        self.logger['Result'].debug('The number of user-item pair in train dataset is {0}'.format(len(self.train_tensor.keys())))

        self.logger['Process'].debug('Get the test dataset')
        self.test_tensor = self.recommender_context.get_data_model().get_data_splitter().get_test_data()
        self.logger['Result'].debug('The number of user-item pair in test dataset is {0}'.format(len(self.test_tensor.keys())))

        self.logger['Process'].debug('Initialize the model parameters')
        self._init_model()

        self.logger['Process'].debug('Building model....')
        self._build_model()

        is_save = self.config_handler['Output', 'is_save', 'bool']
        if is_save:
            self.logger['Process'].debug('Save model ....')
            self._save_model()

        self.logger['Process'].debug('Recommending ...')
        self._recommend()

        self.logger['Process'].debug('Evaluating ...')
        result = self._evaluate()
        self._save_result(result)

        self.logger['Process'].debug("Finish.")
        self.logger['Process'].debug("#"*50)

    def __evaluate_epoch__(self):
        validation_rmse = 0.0
        for user_id, item_id, time_id in self.train_tensor.keys():
            real_rating = self.train_tensor.get((user_id, item_id, time_id))
            predict_rating = self._predict(user_id, item_id, time_id)
            validation_rmse += (real_rating - predict_rating) ** 2
        self._recommend()
        results = self._evaluate()
        return np.sqrt(validation_rmse/len(self.train_tensor.keys())), results['RMSE']

    def _recommend(self):
        for user_id, item_id, time_id in self.test_tensor.keys():
            predict_rating = self._predict(user_id, item_id, time_id) + self.predictions[user_id, item_id, time_id] * self.markov_num
            self.predictions[user_id, item_id, time_id] = predict_rating / (self.markov_num + 1)
        self.markov_num += 1

    # Update hyper-parameters of user or item
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

    # Update time hyper-parameters
    def _sampling_time_hyperparameters(self, factors, normal_dist_mu0, normal_dist_beta0, Wishart_dist_nu0, Wishart_dist_W0):
        num_K = factors.shape[0]
        mu_post = (normal_dist_beta0 * normal_dist_mu0 + factors[0, :]) / (1.0 + normal_dist_beta0)
        beta_post = normal_dist_beta0 + 1.0
        nu_post = Wishart_dist_nu0 + num_K
        X = np.array([factors[t, :] - factors[t-1, :] for t in range(1, num_K)])
        variance_S = np.dot(X.transpose(), X)

        mu0_minus_factors = factors[0, :] - normal_dist_mu0
        mu0_minus_factors = np.reshape(mu0_minus_factors, newshape=(mu0_minus_factors.shape[0], 1))
        W0_post = np.linalg.inv(Wishart_dist_W0) + variance_S + normal_dist_beta0 / (1.0 + normal_dist_beta0) * np.dot((mu0_minus_factors, mu0_minus_factors.transpose()))
        normal_Wishart_distribution = NormalWishartDistribution(mu_post, beta_post, nu_post, W0_post)
        mu, sigma = normal_Wishart_distribution.sample()
        return mu, sigma

    def _update_parameters(self, factors0, factors1, ratings, factors_mu, factors_variance):
        """
        :param factors0:
        :param factors1:
        :param ratings:
        :param factors_mu:
        :param factors_variance:
        :return:
        """
        index = ratings.keys()

        QQ = 0
        RQ = 0
        for dim0, dim1 in index:
            Q = factors0[dim0, :] * factors1[dim1, :]
            QQ += np.mat(Q).transpose() * np.mat(Q)
            RQ += (ratings[dim0, dim1] - self.mean_rating) * Q
        sigma_inv = np.linalg.inv(factors_variance + self.rating_sigma * QQ)
        mu = sigma_inv * (np.dot(factors_variance, np.reshape(factors_mu, newshape=(factors_mu.shape[0], 1))) + self.rating_sigma * RQ)
        return np.random.multivariate_normal(mu, sigma_inv)

    def _update_time_parameters(self, user_factors, item_factors, time_factors, ratings, factors_mu, factors_variance, time_id):
        index = ratings.keys()
        QQ, RQ = 0.0, 0.0
        for dim0, dim1 in index:
            Q = user_factors[dim0, :] * item_factors[dim1, :]
            QQ += np.mat(Q).transpose() * np.mat(Q)
            RQ += (ratings[dim0, dim1] - self.mean_rating) * Q

        RQ = np.reshape(RQ, newshape=(RQ.shape[0], 1))
        if time_id == 0:
            mu = (time_factors[1, :] + factors_mu) / 2
            sigma_inv = np.linalg.inv(2 * factors_variance + self.rating_sigma * QQ)
        elif time_id == self.time_num-1:
            sigma_inv = np.linalg.inv(factors_variance + self.rating_sigma * QQ)
            Tk_1 = np.reshape(time_factors[self.time_num-2, :], newshape=(time_factors.shape[1], 1))
            mu = sigma_inv * (np.dot(factors_variance, Tk_1) + self.rating_sigma * RQ)
        else:
            sigma_inv = np.linalg.inv(2 * factors_variance + self.rating_sigma * QQ)
            Tk = time_factors[time_id-1, :] + time_factors[time_id+1, :]
            mu = sigma_inv * (np.dot(factors_variance, np.reshape(Tk, newshape=(Tk.shape[0], 1))) + self.rating_sigma * RQ)

        return np.random.multivariate_normal(mu, sigma_inv)

    def _predict(self, user_id, item_id, time_id=0):
        assert(time_id < self.time_num)
        predict_rating = np.sum(self.user_factors[user_id, :] * self.item_factors[item_id, :] * self.time_factors[time_id, :]) + self.mean_rating
        if predict_rating > 5:
            return 5
        elif predict_rating < 1:
            return 1
        else:
            return predict_rating

    def _save_result(self, result):
        self.logger['Result'].debug('factor_num: {0}'.format(self.factor_num))

        self.logger['Result'].debug('user_normal_dist_mu0: {0}'.format(self.user_normal_dist_mu0_init))
        self.logger['Result'].debug('user_normal_dist_beta0: {0}'.format(self.user_normal_dist_beta0_init))
        self.logger['Result'].debug('user_Wishart_dist_W0: {0}'.format(self.user_Wishart_dist_W0_init))

        self.logger['Result'].debug('item_normal_dist_mu0: {0}'.format(self.item_normal_dist_mu0_init))
        self.logger['Result'].debug('item_normal_dist_beta0: {0}'.format(self.item_normal_dist_beta0_init))
        self.logger['Result'].debug('item_Wishart_dist_W0: {0}'.format(self.item_Wishart_dist_W0_init))

        self.logger['Result'].debug('time_normal_dist_mu0: {0}'.format(self.time_normal_dist_mu0_init))
        self.logger['Result'].debug('time_normal_dist_beta0: {0}'.format(self.time_normal_dist_beta0_init))
        self.logger['Result'].debug('time_Wishart_dist_W0: {0}'.format(self.time_Wishart_dist_W0_init))

        self.logger['Result'].debug('rating_sigma: {0}'.format(self.rating_sigma_init))
        Recommender._save_result(self, result)

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