#!usr/bin/env python
#coding:utf-8

"""
Reference code: http://www.utstat.toronto.edu/~rsalakhu/code_BPMF/pmf.m
Reference paper: https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf
    momentum: https://en.wikipedia.org/wiki/Stochastic_gradient_descent
"""

import numpy as np
from random import shuffle
from GraphicalRecommender import Recommender


class ProbabilisticMatrixFactorization(Recommender):
    def __init__(self, config_handler):
        Recommender.__init__(self, config_handler)
        # DataModel.__init__(self, config_handler)

    def _read_config(self):
        self.dataset_file = self.config_handler.get_parameter_string('Dataset', 'ratings')
        self.max_iterations = self.config_handler.get_parameter_int('Parameters', 'max_iterations')
        self.factor_num = self.config_handler.get_parameter_int('Parameters', 'factor_num')
        self.learn_rate = self.config_handler.get_parameter_float('Parameters', 'learn_rate')
        self.momentum = self.config_handler.get_parameter_float('Parameters', 'momentum')
        self.user_lambda = self.config_handler.get_parameter_float('Parameters', 'user_lambda')
        self.item_lambda = self.config_handler.get_parameter_float('Parameters', 'item_lambda')
        self.stop_threshold = self.config_handler.get_parameter_float('Parameters', 'stop_threshold')
        self.batch_num = self.config_handler.get_parameter_int('Parameters', 'batch_num')
        self.is_save = self.config_handler.get_parameter_bool('Output', 'is_save')
        self.is_load = self.config_handler.get_parameter_bool('Output', 'is_load')

        self.logger['Result'].debug('max_iterations: {0}'.format(self.max_iterations))
        self.logger['Result'].debug('factor_num: {0}'.format(self.factor_num))
        self.logger['Result'].debug('user_lambda: {0}'.format(self.user_lambda))
        self.logger['Result'].debug('item_lambda: {0}'.format(self.item_lambda))
        self.logger['Result'].debug('learn_rate: {0}'.format(self.learn_rate))
        self.logger['Result'].debug('batch_num: {0}'.format(self.batch_num))
        self.logger['Result'].debug('momentum: {0}'.format(self.momentum))

    def _init_model(self):
        self.user_num, self.item_num = self.train_matrix.shape
        self.record_num = len(self.train_matrix.keys())
        self.rating_mean = self.train_matrix.sum() / self.record_num

        if self.is_load:
            self._load_model()
        else:
            self.user_factors = np.random.normal(0, 1, size=(self.user_num, self.factor_num)) * 0.1
            self.item_factors = np.random.normal(0, 1, size=(self.item_num, self.factor_num)) * 0.1
        self.user_factors_inc = np.zeros((self.user_num, self.factor_num))
        self.item_factors_inc = np.zeros((self.item_num, self.factor_num))

    def _build_model(self):
        losses = list()
        index = np.arange(self.record_num)
        user_item = self.train_matrix.keys()
        users = np.array([user_id for user_id, item_id in user_item])
        items = np.array([item_id for user_id, item_id in user_item])
        ratings = np.array(self.train_matrix.values())
        batch_size = int(self.record_num / self.batch_num)
        for iteration in range(self.max_iterations):
            shuffle(index)

            for batch_id in range(self.batch_num):
                batch_index = index[batch_id*batch_size:(batch_id+1)*batch_size]
                batch_users = users[batch_index]
                batch_items = items[batch_index]
                batch_ratings = ratings[batch_index] - self.rating_mean
                batch_user_factors = self.user_factors[batch_users, :]
                batch_item_factors = self.item_factors[batch_items, :]

                # Compute Prediction
                batch_predictions = np.sum(batch_user_factors * batch_item_factors, axis=-1)
                batch_error = batch_predictions - batch_ratings
                # batch_loss = np.sum(batch_error, batch_error)
                # batch_loss += 0.5 * self.user_lambda * np.sum(np.dot(batch_user_factors, batch_user_factors))
                # batch_loss += 0.5 * self.item_lambda * np.sum(np.dot(batch_item_factors, batch_item_factors))

                # Compute Gradient
                batch_user_delta = \
                    batch_error[..., np.newaxis] * batch_item_factors + self.user_lambda * batch_user_factors
                batch_item_delta = \
                    batch_error[..., np.newaxis] * batch_user_factors + self.item_lambda * batch_item_factors

                user_delta = np.zeros((self.user_num, self.factor_num))
                item_delta = np.zeros((self.item_num, self.factor_num))
                for i in range(batch_size):
                    user_delta[batch_users[i], :] += batch_user_delta[i, :]
                    item_delta[batch_items[i], :] += batch_item_delta[i, :]

                # Update Parameters
                self.user_factors_inc = \
                    self.momentum * self.user_factors_inc + self.learn_rate * user_delta
                self.user_factors -= self.user_factors_inc

                self.item_factors_inc = \
                    self.momentum * self.item_factors_inc + self.learn_rate * item_delta
                self.item_factors -= self.item_factors_inc

                batch_predictions = \
                    np.sum(self.user_factors[batch_users, :] * self.item_factors[batch_items, :], axis=-1)
                batch_error = batch_predictions - batch_ratings
                batch_loss = np.dot(batch_error, batch_error)
                # batch_loss += 0.5 * self.user_lambda * np.sum(
                #     self.user_factors[batch_users, :] * self.user_factors[batch_users, :])
                # batch_loss += 0.5 * self.item_lambda * np.sum(
                #     self.item_factors[batch_items, :] * self.item_factors[batch_items, :])
                losses.append(batch_loss / batch_size)
                self._recommend()
                result = self._evaluate()
                self.logger['Process'].debug("Epoch {0} batch {1}: Training RMSE - {2}, Testing RMSE - {3}".format(
                    iteration, batch_id, losses[-1], result['RMSE']))

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

            self.logger['Process'].debug('Save model')
            if self.is_save:
                self._save_model()

            self.logger['Process'].debug('Prediction.')
            self._recommend()
            self.logger['Process'].debug('Evaluation.')
            result = self._evaluate()
            for key in result:
                self.logger['Result'].debug("{0}: {1} {2}".format(iteration, key, result[key]))

    def _predict(self, user_id, item_id, time_id=0):
        predict_rating = np.dot(self.user_factors[user_id, :], self.item_factors[item_id, :]) + self.rating_mean
        if predict_rating > 5:
            return 5
        elif predict_rating < 1:
            return 1
        else:
            return predict_rating