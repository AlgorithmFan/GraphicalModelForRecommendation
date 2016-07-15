#!usr/bin/env python
#coding:utf-8

from scipy.sparse import dok_matrix
from evaluator.Predict import RMSE, MAE, MSE

class Recommender:
    def __ini__(self, train_matrix, test_matrix, config_handler):
        self.train_matrix = train_matrix
        self.test_matrix = test_matrix
        self.config_handler = config_handler

    def _read_config(self):
        self.max_iterations = self.config_handler.get_parameter_int('PMF', 'max_iterations')

    def _init_model(self):
        self.num_users, self.num_items = self.train_matrix.shape()
        self.predictions = dok_matrix((self.num_users, self.num_items))

    def _build_model(self):
        pass

    def _predict(self):
        pass

    def _evaluate(self):
        mae = MAE.ComputeMeanAbsoluteError(self.prediction, self.testMatrix)
        mse = MSE.ComputeMeanSquareError(self.prediction, self.testMatrix)
        rmse = RMSE.ComputeRootMeanSquareError(self.prediction, self.testMatrix)
        evaluation = {'MAE': mae, 'MSE': mse, 'RMSE': rmse}
        return evaluation

    def run(self):
        self._read_config()
        self._init_model()
        self._build_model()
        self._predict()
        self._evaluate()