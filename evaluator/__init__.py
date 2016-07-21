#!usr/bin/env python
#coding:utf-8

from rating import MAE, RMSE, MSE


class Rating:
    def __init__(self, recommendation, test_matrix):
        self.recommendation = recommendation
        self.test_matrix = test_matrix
        self.evaluator = {'MAE': MAE, 'RMSE': RMSE, 'MSE': MSE}

    def __getitem__(self, item):
        assert (item in self.evaluator)
        return self.evaluator[item](self.recommendation, self.test_matrix)


class Ranking:
    def __init__(self, recommendation, test_matrix):
        self.recommendation = recommendation
        self.test_matrix = test_matrix
        self.evaluator = {}

    def __getitem__(self, item):
        assert (item in self.evaluator)
        return self.evaluator[item](self.recommendation, self.test_matrix)


class Evaluator:
    def __init__(self, recommendation, test_matrix):
        self.rating = Rating(recommendation, test_matrix)
        self.ranking = Ranking(recommendation, test_matrix)

    def __getattr__(self, key):
        if key == 'rating':
            return self.rating
        elif key == 'ranking':
            return self.ranking
        else:
            raise AttributeError


if __name__ == '__main__':
    from scipy.sparse import dok_matrix

    recommendation = dok_matrix((3, 4))
    recommendation[0, 0] = 3
    recommendation[0, 1] = 4
    test_matrix = dok_matrix((3, 4))

    evaluator = Evaluator(recommendation, test_matrix)
    print evaluator.rating['MAE']
    print evaluator.rating['RMSE']
    print evaluator.rating['MSE']