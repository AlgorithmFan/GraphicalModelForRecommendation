#!usr/bin/env python
#coding:utf-8

import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import codecs
import re
import numpy as np
from scipy.sparse import dok_matrix

from util.dateconvert import DateConvertor
from data.sparsetensor import SparseTensor
from DataConvertor import DataConvertor


class TimeDataConvertor (DataConvertor):
    def __init__(self):
        DataConvertor.__init__(self)
        self.data_structure = SparseTensor

    def read_data(self, filename, time_format="month"):
        """
        Read data: (user_id, item_id, rating, timestamp)
        """
        users, items, times = set(), set(), set()
        ratings = list()
        with codecs.open(filename, mode="r", encoding="utf-8") as read_file:
            for line in read_file:
                user_item_timestamp_rating = re.split('\t|,|::', line.strip())
                user_id = int(user_item_timestamp_rating[0])
                item_id = int(user_item_timestamp_rating[1])
                rating = int(user_item_timestamp_rating[2])
                time_id = DateConvertor.convert_timestamp(int(user_item_timestamp_rating[3]), time_format)
                users.add(user_id)
                items.add(item_id)
                times.add(time_id)
                ratings.append((user_id, item_id, time_id, rating))

        # Convert
        user_num, item_num, time_num = len(users), len(items), len(times)
        self.shape = (user_num, item_num, time_num)
        users_dict = {user_id: index for index, user_id in enumerate(list(users))}
        items_dict = {item_id: index for index, item_id in enumerate(list(items))}
        times_dict = {time_id: index for index, time_id in enumerate(list(np.sort(list(times))))}
        sparse_tensor = SparseTensor(shape=(user_num, item_num, time_num))
        for user_id, item_id, time_id, rating in ratings:
            sparse_tensor[users_dict[user_id], items_dict[item_id], times_dict[time_id]] = rating
        self.data = sparse_tensor

    def tensor_matrix(self, tensor_data):
        user_num, item_num = tensor_data.shape[0], tensor_data.shape[1]
        matrix_data = dok_matrix((user_num, item_num))
        for user_id, item_id, time_id in tensor_data.keys():
            matrix_data[user_id, item_id] += tensor_data.get((user_id, item_id, time_id))
        return matrix_data




if __name__ == '__main__':
    file_path = 'D:/Study/Dataset/MovieLens/ml-1m/ratings.dat'
    convertor = TimeDataConvertor()
    # data_model = convertor.read_data(file_path)
    # print 'the number of users is {0}.'.format(data_model.shape[0])
    # print 'the number of items is {0}.'.format(data_model.shape[1])
    # del data_model

    data_model = convertor.read_data(file_path)
    print 'the number of users is {0}'.format(data_model.shape[0])
    print 'the number of items is {0}'.format(data_model.shape[1])
    print 'the number of times is {0}'.format(data_model.shape[2])
    print 'the number of records is {0}'.format(len(data_model.keys()))

    data_matrix = convertor.tensor_matrix(data_model)
    print 'the number of users is {0}'.format(data_matrix.shape[0])
    print 'the number of items is {0}'.format(data_matrix.shape[1])
    print 'the number of records is {0}'.format(len(data_matrix.keys()))