#!usr/bin/env python
#coding:utf-8


def calculateAIC(numParameters, likelihood):
    return 2*numParameters - 2*likelihood



if __name__ == '__main__':
    def hmm(numStates, numItems):
        return numStates * numStates + numStates * numItems + numStates + 2 * numStates

    def ihmm(numStates, numItems):
        return numStates * numStates + numStates * numItems + numStates + 2 * numStates + numStates


    numStates = 40
    numItems = 1621
    numParameters = hmm(numStates, numItems)
    print 'HMM: numStates-{}, likelihood-{}'.format(numStates, calculateAIC(numParameters, -1353854))

    numStates = 30
    numItems = 1621
    numParameters = hmm(numStates, numItems)
    print 'HMM: numStates-{}, likelihood-{}'.format(numStates, calculateAIC(numParameters, -1385505))

    numParameters = ihmm(numStates, numItems)
    print 'IHMM: {}'.format(calculateAIC(numParameters, -2337670))

    print hmm(10, numItems)
    print hmm(20, numItems)
    print hmm(30, numItems)
    print hmm(10, numItems) - hmm(30, numItems)