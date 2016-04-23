#!usr/bin/env python
#coding:utf-8

'''
Paper: Prem Gopalan, et al. Scalable Recommendation with Poisson Factorization.
Author: Haidong Zhang
Date: April 24, 2016
'''

from Recommender import Recommender
import numpy as np
from scipy.sparse import dok_matrix
from util import normalize


class BPoissMF(Recommender):
    def __init__(self, trainMatrix, testMatrix, configHandler):
        Recommender.__init__(trainMatrix, testMatrix, configHandler)





if __name__ == '__main__':
    bnprec = BPoissMF()