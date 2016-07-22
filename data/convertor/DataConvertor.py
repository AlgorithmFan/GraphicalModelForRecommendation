#!usr/bin/env python
#coding:utf-8


class DataConvertor(object):
    def __init__(self):
        self.data = None
        self.shape = None
        self.data_structure = None

    def read_data(self, filename):
        """
        read raw dataset, and convert to sparse matrix format.
        :param filename:
        """
        pass

    def read_given_train_test(self, train_file, test_file):
        """
        read given data set
        """
