#!usr/bin/env python
#coding:utf-8

from GraphicalRecommender import Recommender


class CoDynamicBPMF(Recommender):
    def __init__(self, config_handler):
        Recommender.__init__(self, config_handler)

    def _read_config(self):
        pass

    def _init_model(self):
        pass

    def _build_model(self):
        pass

    def _predict(self, user_id, item_id, time_id=0):
        pass
