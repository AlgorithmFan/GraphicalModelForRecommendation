#!usr/bin/env python
#coding:utf-8

from util.readconf import ReadConfig
from cf_rating import PMF, BPMF, BPTF

if __name__ == '__main__':
    algorithms = {1: PMF, 2: BPMF, 3: BPTF,}
    config_files = {1:"PMF", 2:"BPMF", 3: "BPTF"}
    while True:
        print "0:Exist; 1. PMF; 2. BPMF; 3. BPTF;"
        algorithm_name = input("Please input the algorithm:\n")
        if algorithm_name == 0:
            exit()
        elif algorithm_name in algorithms:
            break
        print "Error, please input correct algorithm name."

    config_file = "config/{0}.cfg".format(config_files[algorithm_name])
    config_handler = ReadConfig(config_file)
    algorithm = algorithms[algorithm_name](config_handler)
    algorithm.run()