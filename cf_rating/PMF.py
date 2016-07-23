#!usr/bin/env python
#coding:utf-8

"""
Reference code: http://www.utstat.toronto.edu/~rsalakhu/code_BPMF/pmf.m
Reference paper: https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf
    momentum: https://en.wikipedia.org/wiki/Stochastic_gradient_descent
"""

import codecs
import numpy as np
from scipy.sparse import dok_matrix
from random import shuffle
from GraphicalRecommender import Recommender


class ProbabilisticMatrixFactorization(Recommender):
    def __init__(self, recommender_context):
        Recommender.__init__(self, recommender_context)

    def _init_model(self):
        self.user_num, self.item_num = self.train_matrix.shape
        self.rating_mean = np.mean(self.train_matrix.values())
        self.predictions = dok_matrix((self.user_num, self.item_num))

        if self.config_handler['Output', 'is_load', 'bool']:
            self._load_model()
            assert(self.user_factors.shape[1] == self.item_factors.shape[1])
            self.factor_num = self.user_factors.shape[1]
        else:
            self.factor_num = self.config_handler['Parameters', 'factor_num', 'int']
            self.user_factors = np.random.normal(0, 1, size=(self.user_num, self.factor_num)) * 0.1
            self.item_factors = np.random.normal(0, 1, size=(self.item_num, self.factor_num)) * 0.1

            # Other Parameters
            self.learn_rate = self.config_handler['Parameters', 'learn_rate', 'float']
            self.momentum = self.config_handler['Parameters', 'momentum', 'float']
            self.user_lambda = self.config_handler['Parameters', 'user_lambda', 'float']
            self.item_lambda = self.config_handler['Parameters', 'item_lambda', 'float']

        # Momentum for update factors
        self.user_factors_inc = np.zeros((self.user_num, self.factor_num))
        self.item_factors_inc = np.zeros((self.item_num, self.factor_num))

    def _build_model(self):

        user_item_keys = self.train_matrix.keys()
        users = np.array([user_id for user_id, item_id in user_item_keys])
        items = np.array([item_id for user_id, item_id in user_item_keys])
        ratings = np.array(self.train_matrix.values())

        # get the index of user_item_keys for stostic
        index = np.arange(len(user_item_keys))
        batch_size = self.config_handler.get_parameter_int('Parameters', 'batch_size')
        batch_num = int(float(len(index)) / batch_size)

        # building model
        losses = list()
        max_iterations = self.config_handler.get_parameter_int('Parameters', 'max_iterations')
        for iteration in range(max_iterations):
            shuffle(index)

            for batch_id in range(batch_num):
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

    def _save_result(self, result):
        self.logger['Result'].debug('factor_num: {0}'.format(self.factor_num))
        self.logger['Result'].debug('learn_rate: {0}'.format(self.learn_rate))
        self.logger['Result'].debug('user_lambda: {0}'.format(self.user_lambda))
        self.logger['Result'].debug('item_lambda: {0}'.format(self.item_lambda))
        self.logger['Result'].debug('momentum: {0}'.format(self.momentum))
        Recommender._save_result(self, result)

    def _predict(self, user_id, item_id, time_id=0):
        predict_rating = np.dot(self.user_factors[user_id, :], self.item_factors[item_id, :]) + self.rating_mean
        if predict_rating > 5:
            return 5
        elif predict_rating < 1:
            return 1
        else:
            return predict_rating

    def _save_model(self):
        save_path = self.config_handler.get_parameter_string("Output", "save_path")
        save_file = save_path + "PMF_{0}.txt".format(self.recommender_context.experiment_id)

        with codecs.open(save_file, mode='w', encoding='utf-8') as write_fp:
            write_fp.write('factor_num: {0}\n'.format(self.factor_num))
            write_fp.write('learn_rate: {0}\n'.format(self.learn_rate))
            write_fp.write('user_lambda: {0}\n'.format(self.user_lambda))
            write_fp.write('item_lambda: {0}\n'.format(self.item_lambda))
            write_fp.write('momentum: {0}\n'.format(self.momentum))
            write_fp.write('user_factors \n')
            self._save_matrix(self.user_factors, write_fp)

            write_fp.write('item_factors \n')
            self._save_matrix(self.item_factors, write_fp)

    def _load_model(self):
        load_path = self.config_handler.get_parameter_string("Output", "load_path")
        load_file = load_path + "PMF_{0}.txt".format(self.recommender_context.experiment_id)

        with codecs.open(load_file, mode='r', encoding='utf-8') as read_fp:
            for line in read_fp:
                if line.startswith('factor_num'):
                    self.factor_num = int(line.split(':')[1].strip())
                elif line.startswith('learn_rate'):
                    self.learn_rate = float(line.split(':')[1].strip())
                elif line.startswith('user_lambda'):
                    self.user_lambda = float(line.split(':')[1].strip())
                elif line.startswith('item_lambda'):
                    self.item_lambda = float(line.split(':')[1].strip())
                elif line.startswith('momentum'):
                    self.momentum = float(line.split(':')[1].strip())
                elif line.startswith('user_factor'):
                    self.user_factors = self._load_matrix(read_fp)
                elif line.startswith('item_factor'):
                    self.item_factors = self._load_matrix(read_fp)