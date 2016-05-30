#!usr/bin/env python
#coding:utf-8


def calculateAIC(numParameters, likelihood):
    return 2*numParameters - 2*likelihood



if __name__ == '__main__':
    def hmm(numStates, numItems):
        return numStates * numStates + numStates * numItems + numStates + 2 * numStates

    def ihmm(numStates, numItems):
        return numStates * numStates + numStates * numItems + numStates + 2 * numStates + numStates


    numStates = 20
    numItems = 1621
    numParameters = hmm(numStates, numItems)
    print 'HMM: {}'.format(calculateAIC(numParameters, -2815350))

    numStates = 30
    numItems = 1621
    numParameters = hmm(numStates, numItems)
    print 'HMM: {}'.format(calculateAIC(numParameters, -2761384))

    numParameters = ihmm(numStates, numItems)
    print 'IHMM: {}'.format(calculateAIC(numParameters, -2337670))