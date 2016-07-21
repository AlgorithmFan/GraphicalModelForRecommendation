#!usr/bin/env python
#coding:utf-8

import numpy as np


class SparseTensor:
    def __init__(self, shape):
        self.data = dict()
        self.shape = shape

    def __setitem__(self, key, value):
        assert(len(key) == len(self.shape))
        for i in range(len(self.shape)):
            assert(key[i] < self.shape[i])
        self.data[key] = value

    def __getitem__(self, item):
        assert(len(item) == len(self.shape))
        for i in range(len(self.shape)):
            assert(item[i] < self.shape[i])
        return self.get(item)

    def keys(self):
        return self.data.keys()

    def get(self, key):
        if key in self.data:
            return self.data[key]
        else:
            return 0

    def get_dimension(self, dim=0, value=0):
        index = [i for i in range(dim)]
        index.extend([i for i in range(dim+1, len(self.shape))])
        shape = tuple(np.array(self.shape)[index])
        data = dict()

        for key in self.keys():
            if key[dim] != value:
                continue
            _key = np.array(key)[index]
            data[tuple(_key)] = self.data[key]
        t = SparseTensor(shape)
        t.data = data
        return t

if __name__ == '__main__':
    tensor = SparseTensor(shape=(3, 4, 4))
    print tensor.keys()
    tensor[0, 1, 2] = 1
    tensor[1, 3, 2] = 4
    tensor[1, 2, 3] = 2
    print tensor.keys()
    print tensor.get((0, 1, 2))
    print tensor[1, 3, 2]
    tensor = tensor.get_dimension(dim=0, value=1)
    print tensor.shape
    print tensor.keys()
    print tensor[1, 1]