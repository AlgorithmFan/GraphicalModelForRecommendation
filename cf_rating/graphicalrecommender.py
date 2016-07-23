#!usr/bin/env python
#coding:utf-8

from scipy.sparse import dok_matrix
from evaluator import Evaluator
import numpy as np


class Recommender:
    def __init__(self, recommender_context):
        self.train_data = None
        self.test_data = None
        self.recommender_context = recommender_context
        self.config_handler = self.recommender_context.get_config()
        self.logger = self.recommender_context.get_logger()

    def _init_model(self):
        self.user_num, self.item_num = self.train_matrix.shape
        self.mean_rating = np.mean(self.train_matrix.values())

        self.predictions = dok_matrix((self.user_num, self.item_num))

        self.factor_num = self.config_handler.get_parameter_int('Parameter', 'factor_num')
        self.user_factors = np.random.normal(0, 1, size=(self.user_num, self.factor_num)) * 0.1
        self.item_factors = np.random.normal(0, 1, size=(self.item_num, self.factor_num)) * 0.1
        self.user_factors_inc = np.zeros((self.user_num, self.factor_num))
        self.item_factors_inc = np.zeros((self.item_num, self.factor_num))

    def _build_model(self):
        self.max_iterations = self.config_handler.get_parameter_int('Parameter', 'max_iterations')

    def _recommend(self):
        for user_id, item_id in self.test_matrix.keys():
            self.predictions[user_id, item_id] = self._predict(user_id, item_id)

    def _predict(self, user_id, item_id, time_id=0):
        return 0.0

    def _evaluate(self):
        evaluator_cfg = self.config_handler.get_parameter_string("Output", 'evaluator')
        evaluator_cfg = evaluator_cfg.strip().split(',')
        evaluator = Evaluator(self.predictions, self.test_matrix)
        result = {}
        for key in evaluator_cfg:
            result[key] = evaluator.rating[key.strip()]
        return result

    def run(self):
        self.logger['Process'].debug('Get the train dataset')
        self.train_matrix = self.recommender_context.get_data_model().get_data_splitter().get_train_matrix()
        self.logger['Result'].debug('The number of user-item pair in train dataset is {0}'.format(len(self.train_matrix.keys())))

        self.logger['Process'].debug('Get the test dataset')
        self.test_matrix = self.recommender_context.get_data_model().get_data_splitter().get_test_matrix()
        self.logger['Result'].debug('The number of user-item pair in test dataset is {0}'.format(len(self.test_matrix.keys())))

        self.logger['Process'].debug('Initialize the model parameters')
        self._init_model()

        self.logger['Process'].debug('Building model....')
        self._build_model()

        is_save = self.config_handler.get_parameter_bool('Output', 'is_save')
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

    def _save_result(self, result):
        for key in result:
            self.logger['Result'].debug("{0}: {1}".format(key, result[key]))

    def _save_model(self):
        pass

    def _load_model(self):
        pass

    def _load_matrix(self, read_fp):
        data = list()
        for vector in read_fp:
            if vector.startswith('matrix_end'):
                break
            vector = vector.strip().split('\t')
            vector = [float(feature) for feature in vector]
            data.append(vector)
        return np.array(data)

    def _save_matrix(self, matrix, write_fp):
        for vector in matrix:
            for feature in vector:
                write_fp.write("{0}\t".format(feature))
            write_fp.write("\n")
        write_fp.write('matrix_end\n')
