#!usr/bin/env python
#coding:utf-8


class DataSplitter:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.validation_data = None
        self.convertor = None

    def set_data_convertor(self, data_convertor):
        self.data_convertor = data_convertor

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_validation_data(self):
        return self.validation_data
