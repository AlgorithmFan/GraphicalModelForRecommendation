#!usr/bin/env python
#coding:utf-8

import splitter
import convertor


class DataModel:
    def __init__(self, config_handler):
        self.config_handler = config_handler

    def build_data_model(self):
        """
        Read data to convertor, and initialize splitter
        """
        data_format = self.config_handler.get_parameter_string('Dataset', 'data_format')

        # Read data to convertor
        if data_format == 'time':
            self.convertor = convertor.TimeDataConvertor()
        elif data_format == 'document':
            self.convertor = convertor.DocumentDataConvertor()
        else:
            self.convertor = convertor.GeneralDataConvertor()
        dataset_file = self.config_handler.get_parameter_string('Dataset', 'ratings')
        self.convertor.read_data(dataset_file)

        # Initialize splitter, and transport convertor into the splitter
        splitter_method = self.config_handler.get_parameter_string('splitter', 'method')
        splitter_method_index = self.config_handler.get_parameter_int('splitter', 'method_index')
        splitter_method_parameter = self.config_handler.get_parameter_float('splitter', 'method_parameter')
        if splitter_method == 'given_n':
            self.splitter = splitter.GivenNDataSplitter(self.convertor, splitter_method_index, splitter_method_parameter)
        elif splitter_method == 'generic':
            self.splitter = splitter.GenericDataSplitter(self.convertor, splitter_method_index, splitter_method_parameter)
        elif splitter_method == 'ratio':
            self.splitter = splitter.GenericDataSplitter(self.convertor, splitter_method_index, splitter_method_parameter)
        elif splitter_method == 'cv':
            self.splitter = splitter.CrossValidationDataSplitter(self.convertor, splitter_method_index, splitter_method_parameter)

    def get_data_splitter(self):
        return self.splitter

    def get_data_convertor(self):
        return self.convertor


if __name__ == '__main__':
    pass