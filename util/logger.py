#!usr/bin/env python
#coding:utf-8


import logging
from logging.handlers import RotatingFileHandler
# import logging.config


class Logger:
    def __init__(self, filename):
        self.filename = filename
        self.logger = None

    def _set_config(self):
        pass

    def _set_console(self):
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        self.logger.addHandler(console)

    def information(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def warn(self, message):
        self.logger.warn(message)

    def error(self, message):
        self.logger.error(message)


class Process(Logger):
    def __init__(self, filename):
        Logger.__init__(self, filename)
        self._set_config()
        self._set_console()

    def _set_config(self):
        Rthandler = RotatingFileHandler(filename=self.filename,
                                        maxBytes=10*1024*1024,
                                        backupCount=5, mode='a')
        Rthandler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        Rthandler.setFormatter(formatter)
        self.logger = logging.getLogger('Progress')
        self.logger.addHandler(Rthandler)
        self.logger.setLevel(logging.DEBUG)


class Result(Logger):
    def __init__(self, filename):
        Logger.__init__(self, filename)
        self._set_config()
        self._set_console()

    def _set_config(self):
        handler = logging.FileHandler(filename=self.filename, mode='a')
        frt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler.setFormatter(frt)
        handler.setLevel(logging.DEBUG)
        self.logger = logging.getLogger('Result')
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    logger = {'Result': Result('../output/result.log'), 'Process': Process('../output/process.log')}
    logger['Result'].debug('Wrong')
    logger['Process'].debug('Wrong')