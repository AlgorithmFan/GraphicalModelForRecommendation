#!usr/bin/env python
#coding:utf-8


import logging
import logging.config

class Logger:
    def __init__(self, filename):
        self.filename = filename
        logging.basicConfig(filename=self.filename, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    def information(self, str):
        logging.info(str)

    def warn(self, str):
        logging.warn(str)

    def error(self, str):
        logging.error(str)





if __name__ == '__main__':

   logger = Logger('hello.log')
   logger.information('hello')