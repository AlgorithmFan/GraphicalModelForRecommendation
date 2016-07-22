#!usr/bin/env python
#coding:utf-8

from scipy.sparse import dok_matrix
from evaluator import Evaluator
from data.convertor import Convertor
from data.splitter import Splitter
from util import logger
import numpy as np
import codecs


class Recommender:
    def __init__(self, config_handler):
        self.train_matrix = None
        self.test_matrix = None
        self.config_handler = config_handler
        self._set_logger()

    def _set_logger(self):
        result_file = self.config_handler.get_parameter_string("Output", "logger") + "{0}_Result.log".format(self.__class__.__name__)
        process_file = self.config_handler.get_parameter_string("Output", "logger") + "{0}_Process.log".format(self.__class__.__name__)
        self.logger = {'Result': logger.Result(result_file), 'Process': logger.Process(process_file)}

    def _read_config(self):
        self.max_iterations = self.config_handler.get_parameter_int('Parameter', 'max_iterations')
        self.dataset_file = self.config_handler.get_parameter_string('Dataset', 'rating')
        self.factor_num = self.config_handler.get_parameter_int('Parameters', 'factor_num')

    def _get_data_model(self):
        self.convertor = Convertor()
        data = self.convertor.read_tensor(self.dataset_file)
        self.splitter = Splitter(data)

    def _init_model(self):
        self.user_num, self.item_num = self.train_matrix.shape
        self.record_num = len(self.train_matrix.keys())
        self.rating_mean = self.train_matrix.sum() / self.record_num

        self.predictions = dok_matrix((self.user_num, self.item_num))

        self.user_factors = np.random.normal(0, 1, size=(self.user_num, self.factor_num)) * 0.1
        self.item_factors = np.random.normal(0, 1, size=(self.item_num, self.factor_num)) * 0.1
        self.user_factors_inc = np.zeros((self.user_num, self.factor_num))
        self.item_factors_inc = np.zeros((self.item_num, self.factor_num))

    def _build_model(self):
        pass

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
        self.logger['Process'].debug("\n" + "#"*50 + "Start" + "#"*50)
        self.logger['Result'].debug("\n" + "#"*50 + "Start" + "#"*50)
        self.logger['Process'].debug("Read config file.")
        self._read_config()

        self.logger['Process'].debug("Read dataset.")
        self._get_data_model()
        splitter_mode = self.config_handler.get_parameter_string("Dataset", "splitter")

        self.logger['Process'].debug("Split dataset according to {0}".format(splitter_mode))
        if splitter_mode == "time":
            self._run_time_sequence_algorithm()
        elif splitter_mode == "cv":
            self._run_cross_validation()
        elif splitter_mode == "":
            pass

        self.logger['Process'].debug("Finish.")
        self.logger['Process'].debug("#"*50)

    def _run_time_sequence_algorithm(self):
        experiment_num = self.config_handler.get_parameter_int("Dataset", "experiment_num")
        time_num = self.splitter.data_model.shape[2]
        for iteration in range(experiment_num):
            self.logger['Process'].debug('#'*50)
            self.logger['Process'].debug('The {0}-th experiment'.format(iteration))
            self.logger['Process'].debug('Split the dataset.')
            self.train_tensor, self.test_tensor = self.splitter.get_given_n_by_time(iteration, time_num-experiment_num)
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

    def _save_model(self):
        save_path = self.config_handler.get_parameter_string("Output", "save_path")
        save_file = save_path + self.__class__.__name__ + "_user_factors{0}.txt".format(self.experiment)
        self._save_matrix(self.user_factors, save_file)

        save_file = save_path + self.__class__.__name__ + "_item_factors{0}.txt".format(self.experiment)
        self._save_matrix(self.item_factors, save_file)

    def _load_model(self):
        load_path = self.config_handler.get_parameter_string("Output", "load_path")
        load_file = load_path + self.__class__.__name__ + "_user_factors{0}.txt".format(self.experiment)
        self.user_factors = self._load_matrix(load_file)

        load_file = load_path + self.__class__.__name__ + "_item_factors{0}.txt".format(self.experiment)
        self.item_factors = self._load_matrix(load_file)

    def _load_matrix(self, filename):
        data = list()
        with codecs.open(filename, mode='r', encoding='utf-8') as read_fp:
            for vector in read_fp:
                vector = vector.strip().split('\t')
                vector = [float(feature) for feature in vector]
                data.append(vector)
        return np.array(data)

    def _save_matrix(self, matrix, filename):
        with codecs.open(filename, mode='w', encoding='utf-8') as write_fp:
            for vector in matrix:
                for feature in vector:
                    write_fp.write("{0}\t".format(feature))
                write_fp.write("\n")

    def _run_cross_validation(self):
        # ratio = self.config_handler.get_parameter_float("Dataset", "ratio")
        pass