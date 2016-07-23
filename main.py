#!usr/bin/env python
#coding:utf-8

from GMRec import GMRec


if __name__ == '__main__':
    algorithms = {1:"PMF", 2:"BPMF", 3: "BPTF"}
    while True:
        print "0:Exist; 1. PMF; 2. BPMF; 3. BPTF;"
        algorithm_index = input("Please input the algorithm:\n")
        if algorithm_index == 0:
            exit()
        elif algorithm_index in algorithms:
            break
        print "Error, please input correct algorithm name."

    config_file = "config/{0}.cfg".format(algorithms[algorithm_index])
    gmrec = GMRec(config_file, algorithms[algorithm_index])
    gmrec.run()