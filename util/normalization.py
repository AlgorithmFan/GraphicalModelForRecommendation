#!usr/bin/env python
#coding:utf-8

'''
@author: haidong zhang

'''
import numpy as np

def normalize(matrix):
    '''
    Normalize the matrix
    '''
    numDims = len(matrix.shape)
    if numDims == 1:
        # a vector
        s = np.sum(matrix)
        assert(s != 0)
        return matrix / s
    else:
        # a matrix
        s = np.sum(matrix, axis=1)
        return matrix / s[..., np.newaxis]