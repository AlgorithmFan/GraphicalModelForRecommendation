#1usr/bin/env python
#coding:utf-8

import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from data.sparsetensor import SparseTensor
from scipy.sparse import dok_matrix


class Splitter(object):
    def __init__(self, data_model):
        self.data_model = data_model

    def get_given_n_by_time(self, start_time_id, given_time_num):
        user_num, item_num, time_num = self.data_model.shape
        train_tensor = SparseTensor(shape=(user_num, item_num, given_time_num))
        test_tensor = SparseTensor((user_num, item_num, 1))
        for user_id, item_id, time_id in self.data_model.keys():
            if start_time_id <= time_id < start_time_id+given_time_num:
                train_tensor[user_id, item_id, time_id-start_time_id] = self.data_model.get((user_id, item_id, time_id))
            elif time_id == start_time_id+given_time_num:
                test_tensor[user_id, item_id, 0] = self.data_model.get((user_id, item_id, time_id))
        return train_tensor, test_tensor

    def get_ratio_by_rating(self, ratio):
        user_num, item_num = self.data_model.shape
        train_matrix = dok_matrix((user_num, item_num))
        test_matrix = dok_matrix((user_num, item_num))
        for user_id, item_id, time_id in self.data_model.keys():
            pass

    def get_ratio_by_rating_date(self, ratio):
        pass

    def get_ratio_by_user_date(self, ratio):
        pass

    def get_ratio_by_item_date(self, ratio):
        pass

    def get_ratio_by_user(self, ratio):
        pass

    def get_ratio_by_item(self, ratio):
        pass

    def get_ratio(self, ratio):
        pass

    def get_given_n_by_user(self, given_num):

        pass

    def get_given_n_by_item(self, given_num):
        # user_num, item_num, time_num = self.data_model.shape
        # train_tensor = SparseTensor(shape=(user_num, item_num, given_time_num))
        # test_tensor = SparseTensor((user_num, item_num, 1))
        # for user_id, item_id, time_id in self.data_model.keys():
        #     if start_time_id <= time_id < start_time_id+given_time_num:
        #         train_tensor[user_id, item_id, time_id-start_time_id] = self.data_model.get((user_id, item_id, time_id))
        #     elif time_id == start_time_id+given_time_num:
        #         test_tensor[user_id, item_id, 0] = self.data_model.get((user_id, item_id, time_id))
        # return train_tensor, test_tensor
        pass


