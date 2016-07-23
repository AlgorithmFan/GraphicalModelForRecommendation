#!usr/bin/env python
#coding:utf-8

from DataSplitter import DataSplitter


class CrossValidationDataSplitter (DataSplitter):
    def __init__(self, convertor, splitter_method_index, splitter_method_parameter):
        DataSplitter.__init__(self, convertor, splitter_method_index, splitter_method_parameter)