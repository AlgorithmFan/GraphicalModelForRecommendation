#!usr/bin/env python
#coding: utf-8


def MSE(recommendation, test_matrix):
    loss = 0.0
    number = 0
    for key in recommendation.keys():
        if len(key) == 2:
            user_id, item_id = key
        elif len(key) == 3:
            user_id, item_id, time_id = key
        else:
            raise AttributeError

        error = recommendation.get((user_id, item_id)) - test_matrix.get(key)
        loss += error * error
        number += 1
    if number > 0:
        return loss / number
    return 0.0

