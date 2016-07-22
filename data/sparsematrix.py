#!usr/bin/env python
#coding:utf-8

import numpy as np


class SparseMatrix:
    def __init__(self, shape):
        assert(len(shape) == 2)
        self.data = dict()
        self.row_data = dict()
        self.col_data = dict()
        self.shape = shape

    def __setitem__(self, key, value):
        assert(len(key) == 2)
        assert(key[0] < self.shape[0])
        assert(key[1] < self.shape[1])
        self.data[key] = value
        self.row_data.setdefault(key[0], SparseMatrix(shape=(1, self.shape[1])))
        self.row_data[key[0]].data[0, key[1]] = value
        self.col_data.setdefault(key[1], SparseMatrix(shape=(self.shape[0], 1)))
        self.col_data[key[1]].data[key[0], 0] = value

    def __getitem__(self, key):
        assert(len(key) == 2)
        assert(key[0] < self.shape[0])
        assert(key[1] < self.shape[1])
        return self.get(key)

    def keys(self):
        return self.data.keys()

    def get(self, key):
        if key in self.data:
            return self.data[key]
        else:
            return 0

    def getrow(self, row_id):
        if row_id in self.row_data:
            return self.row_data[row_id]
        else:
            return SparseMatrix(shape=(1, self.shape[1]))

    def getcol(self, col_id):
        if col_id in self.col_data:
            return self.col_data[col_id]
        else:
            return SparseMatrix(shape=(self.shape[0], 1))

    def transpose(self):
        sparse_matrix = SparseMatrix(shape=(self.shape[1], self.shape[0]))
        for row_id, col_id in self.keys():
            sparse_matrix[col_id, row_id] = self.get((row_id, col_id))
        return sparse_matrix

    def values(self):
        return self.data.values()


if __name__ == '__main__':
    matrix = SparseMatrix(shape=(3, 4))
    print 'Matrix Shape: {0}'.format(matrix.keys())
    matrix[0, 2] = 1
    matrix[0, 3] = 3
    matrix[1, 2] = 4
    print 'Matrix Shape: {0}'.format(matrix.keys())

    row_matrix = matrix.getrow(0)
    print 'Matrix Shape: {0}'.format(row_matrix.keys())

    col_matrix = matrix.getcol(2)
    print 'Matrix Shape: {0}'.format(col_matrix.keys())

    col_matrix = col_matrix.transpose()
    print 'Matrix Shape: {0}'.format(col_matrix.keys())
