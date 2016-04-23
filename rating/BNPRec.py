#!usr/bin/env python
#coding:utf-8

'''
Paper: Prem Gopalan, Francisco J.R. Ruiz, et al. Bayesian Non-parameter Poisson Factorization for Recommendation Systems
Author: Haidong Zhang
Date: April 16, 2016
'''

from Recommender import Recommender
import numpy as np
from scipy.sparse import dok_matrix
from util import normalize


class BNPRec(Recommender):
    def __init__(self, trainMatrix, testMatrix, configHandler):
        Recommender.__init__(trainMatrix, testMatrix, configHandler)



if __name__ == '__main__':
    bnprec = BNPRec()