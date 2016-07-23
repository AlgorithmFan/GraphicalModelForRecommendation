#!usr/bin/env python
#coding:utf-8


class RecommenderContext:
    def __init__(self, config_handler, data_model, logger):
        self.config_handler = config_handler
        self.data_model = data_model
        self.logger = logger
        self.experiment_id = 0

    def get_config(self):
        return self.config_handler

    def get_data_model(self):
        return self.data_model

    def get_logger(self):
        return self.logger


