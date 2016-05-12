#!usr/bin/env python
#coding:utf-8

'''

'''

import numpy as np
from scipy import stats

if __name__ == '__main__':
    x = [850,740,900,1070,930,850,950,980,980,880,1000,980,930,650,760,810,1000,1000,960,960]
    x = np.array(x)
    x1 = x - 1
    print 't-statistic = %6.3f pvalue = %6.4f' %  stats.ttest_ind(x, x1)
# [h,pvalue,ci]=ttest(x,990)