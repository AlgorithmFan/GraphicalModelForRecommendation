#!usr/bin/env python
#coding:utf-8

from DataSplitter import DataSplitter


class RatioDataSplitter (DataSplitter):
    def __init__(self, convertor, splitter_method_index, splitter_method_parameter):
        DataSplitter.__init__(self, convertor, splitter_method_index, splitter_method_parameter)
        self.splitter_ratio = splitter_method_parameter