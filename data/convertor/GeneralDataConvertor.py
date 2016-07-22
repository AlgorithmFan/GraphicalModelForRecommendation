#!usr/bin/env python
#coding:utf-8

import codecs
import re
from scipy.sparse import dok_matrix


class GeneralDataConvertor(object):
    def __init__(self):
        pass

    def read_data(self, filename):
        """
        read raw dataset, and convert to sparse matrix format.
        :param filename:
        """
        users, items = set(), set()
        ratings = list()
        with codecs.open(filename, mode="r", encoding="utf-8") as read_file:
            for line in read_file:
                user_item_rating = re.split('\t|,|::', line.strip())
                user_id = int(user_item_rating[0])
                item_id = int(user_item_rating[1])
                rating = int(user_item_rating[2])
                users.add(user_id)
                items.add(item_id)
                ratings.append((user_id, item_id, rating))

        # Convert
        user_num, item_num = len(users), len(items)
        users_dict = {user_id: index for index, user_id in enumerate(list(users))}
        items_dict = {item_id: index for index, item_id in enumerate(list(items))}
        data_model = dok_matrix((user_num, item_num))
        for user_id, item_id, rating in ratings:
            data_model[users_dict[user_id], items_dict[item_id]] = rating
        return data_model

    def read_given_train_test(self, train_file, test_file):
        """
        read given data set
        """
        users, items = set(), set()
        ratings = list()
        with codecs.open(train_file, mode="r", encoding="utf-8") as read_file:
            for line in read_file:
                user_item_rating = re.split('\t|,|::', line.strip())
                user_id = int(user_item_rating[0])
                item_id = int(user_item_rating[1])
                rating = int(user_item_rating[2])
                users.add(user_id)
                items.add(item_id)
                ratings.append((user_id, item_id, rating))

        # Convert
        user_num, item_num = len(users), len(items)
        users_dict = {user_id: index for index, user_id in enumerate(list(users))}
        items_dict = {item_id: index for index, item_id in enumerate(list(items))}
        train_matrix = dok_matrix((user_num, item_num))
        test_matrix = dok_matrix((user_num, item_num))
        for user_id, item_id, rating in ratings:
            train_matrix[users_dict[user_id], items_dict[item_id]] = rating

        with codecs.open(test_file, mode='r', encoding='utf-8') as read_file:
            for line in read_file:
                user_item_rating = re.split('\t|,|::', line.strip())
                user_id = int(user_item_rating[0])
                item_id = int(user_item_rating[1])
                rating = int(user_item_rating[2])
                test_matrix[users_dict[user_id], items_dict[item_id]] = rating
        return train_matrix, test_matrix