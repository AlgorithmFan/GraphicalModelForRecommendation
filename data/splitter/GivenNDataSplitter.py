#!usr/bin/env python
#coding:utf-8


from DataSplitter import DataSplitter

import numpy as np


class GivenNDataSplitter(DataSplitter):
    def __init__(self):
        DataSplitter.__init__(self)

    def get_given_n_by_user(self, given_num):
        """
        Split ratings into two parts: training set consisting of user-item ratings where {@code given_num} ratings
        are preserved for each user, and the rest are used as the testing data.
        """
        assert (given_num > 0)
        self.train_data = self.convertor.data_structure(self.convertor.shape)
        self.test_data = self.convertor.data_structure(self.convertor.shape)

        user_keys = dict()
        for key in self.convertor.data.keys():
            user_keys.setdefault(key[0], list())
            user_keys[key[0]].append(key)

        for user_id in user_keys:
            rating_num = len(user_keys[user_id])
            if rating_num > given_num:
                index = np.arange(rating_num)
                np.random.shuffle(index)
                for i in index[:rating_num-given_num]:
                    key = user_keys[user_id][index[i]]
                    self.train_data[key] = self.convertor.data[key]
                for i in index[rating_num-given_num:]:
                    key = user_keys[user_id][index[i]]
                    self.test_data[key] = self.convertor.data[key]
            else:
                for key in user_keys[user_id]:
                    self.test_data[key] = self.convertor.data[key]

    def get_given_n_by_item(self, given_num):
        """
        Split ratings into two parts: training set consisting of user-item ratings where {@code given_num} ratings
        are preserved for each item, and the rest are used as the testing data.
        """
        assert (given_num > 0)
        self.train_data = self.convertor.data_structure(self.convertor.shape)
        self.test_data = self.convertor.data_structure(self.convertor.shape)

        item_keys = dict()
        for key in self.convertor.data.keys():
            item_keys.setdefault(key[1], list())
            item_keys[key[1]].append(key)

        for item_id in item_keys:
            rating_num = len(item_keys[item_id])
            if rating_num > given_num:
                index = np.arange(rating_num)
                np.random.shuffle(index)
                for i in index[:rating_num-given_num]:
                    key = item_keys[item_id][index[i]]
                    self.train_data[key] = self.convertor.data[key]
                for i in index[rating_num-given_num:]:
                    key = item_keys[item_id][index[i]]
                    self.test_data[key] = self.convertor.data[key]
            else:
                for key in item_keys[item_id]:
                    self.test_data[key] = self.convertor.data[key]

    def get_given_n_by_user_date(self, given_num):
        pass

    def get_given_n_by_item_date(self, given_num):
        pass

    def get_data_model(self):
        pass

    def split_data(self):
        pass
