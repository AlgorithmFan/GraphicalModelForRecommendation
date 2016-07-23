#!usr/bin/env python
#coding:utf-8

import ConfigParser


class ReadConfig:
    def __init__(self, config_file_path):
        self.cf = ConfigParser.ConfigParser()
        self.cf.read(config_file_path)

    def __getitem__(self, key):
        assert(len(key) == 3)
        if key[2] == 'string':
            return self.get_parameter_string(key[0], key[1])
        elif key[2] == 'bool':
            return self.get_parameter_bool(key[0], key[1])
        elif key[2] == 'int':
            return self.get_parameter_int(key[0], key[1])
        elif key[2] == 'float':
            return self.get_parameter_float(key[0], key[1])
        else:
            raise KeyError

    def get_parameter_string(self, section, key):
        return self.cf.get(section, key)

    def get_parameter_int(self, section, key):
        return int(self.get_parameter_string(section, key))

    def get_parameter_float(self, section, key):
        return float(self.get_parameter_string(section, key))

    def get_parameter_bool(self, section, key):
        return bool(self.get_parameter_int(section, key))