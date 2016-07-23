#!usr/bin/env python
#coding:utf-8

import cPickle
import os
from scipy.sparse import dok_matrix

class DataSplitter:
    def __init__(self, convertor, splitter_method_index, splitter_method_parameter):
        self.train_data = None
        self.test_data = None
        self.validation_data = None
        self.convertor = convertor
        self.splitter_method_index = splitter_method_index
        self.splitter_mathod_parameter = splitter_method_parameter
        self.methods = dict()

    def set_data_convertor(self, data_convertor):
        self.data_convertor = data_convertor

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_train_matrix(self):
        if len(self.train_data.shape) == 2:
            return self.train_data
        train_matrix = dok_matrix((self.train_data.shape[0], self.train_data.shape[1]))
        for key in self.train_data.keys():
            train_matrix[key[0], key[1]] = self.train_data[key]
        return train_matrix

    def get_test_matrix(self):
        if len(self.test_data.shape) == 2:
            return self.test_data
        test_matrix = dok_matrix((self.test_data.shape[0], self.test_data.shape[1]))
        for key in self.test_data.keys():
            test_matrix[key[0], key[1]] = self.test_data[key]
        return test_matrix

    def get_validation_data(self):
        return self.validation_data

    def split_data(self, save_path, experiment_id):
        self.save_train_test_data(save_path, experiment_id)

    def save_train_test_data(self, save_path, experiment_id):
        save_file = save_path + "train_matrix_{0}.bin".format(experiment_id)
        self._save_data(self.train_data, save_file)

        save_file = save_path + "test_matrix_{0}.bin".format(experiment_id)
        self._save_data(self.test_data, save_file)

    def load_train_test_data(self, load_path, experiment_id):
        load_file = load_path + "train_matrix_{0}.bin".format(experiment_id)
        if os.path.exists(load_file):
            self.train_data = self._load_data(load_file)

        load_file = load_path + "test_matrix_{0}.bin".format(experiment_id)
        if os.path.exists(load_file):
            self.test_data = self._load_data(load_file)
            return True
        return False

    def _save_data(self, data, filename):
        with open(filename, 'w') as write_fp:
            cPickle.dump(data, write_fp)

    def _load_data(self, filename):
        with open(filename, 'r') as read_fp:
            data = cPickle.load(read_fp)
        return data