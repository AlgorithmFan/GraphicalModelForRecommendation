#!usr/bin/env python
#coding:utf-8

import ConfigParser
class ReadConfig:
    def __init__(self, config_file_path):
        self.cf = ConfigParser.ConfigParser()
        self.cf.read(config_file_path)

    def getParameter(self, section, key):
        return self.cf.get(section, key)

