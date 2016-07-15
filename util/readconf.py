#!usr/bin/env python
#coding:utf-8

import ConfigParser

class ReadConfig:
    def __init__(self, config_file_path):
        self.cf = ConfigParser.ConfigParser()
        self.cf.read(config_file_path)

    def get_parameter_string(self, section, key):
        return self.cf.get(section, key)

    def get_parameter_int(self, section, key):
        return int(self.get_parameters_string(section, key))

    def get_parameter_float(self, section, key):
        return float(self.get_parameters_string(section, key))
