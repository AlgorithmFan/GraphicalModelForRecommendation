#!usr/bin/env python
#coding:utf-8


def calculateAIC(numParameters, likelihood):
    return 2*numParameters - 2*likelihood



if __name__ == '__main__':
    def hmm(numStates, numItems):
        return numStates * numStates + numStates * numItems + numStates + 2 * numStates

    def ihmm(numStates, numItems):
        return numStates * numStates + numStates * numItems + numStates + 2 * numStates + numStates


    numStates = 10
    numItems = 1621
    likelihood = -1389871
    numParameters = hmm(numStates, numItems)
    print 'HMM: numStates-{}, numParameters-{}, likelihood-{}'.format(numStates, numParameters, calculateAIC(numParameters, likelihood))

    numStates = 20
    numItems = 1621
    likelihood = -1321175
    numParameters = hmm(numStates, numItems)
    print 'HMM: numStates-{}, numParameters-{}, likelihood-{}'.format(numStates, numParameters, calculateAIC(numParameters, likelihood))

    numStates = 30
    numItems = 1621
    likelihood = -1270278
    numParameters = hmm(numStates, numItems)
    print 'HMM: numStates-{}, numParameters-{}, likelihood-{}'.format(numStates, numParameters, calculateAIC(numParameters, likelihood))

    numStates = 40
    numItems = 1621
    likelihood = -1353854
    numParameters = hmm(numStates, numItems)
    print 'HMM: numStates-{}, numParameters-{},likelihood-{}'.format(numStates, numParameters, calculateAIC(numParameters, likelihood))

    numStates = 10
    numItems = 1621
    likelihood = -1349191
    numParameters = ihmm(numStates, numItems)
    print 'IHMM: numStates-{}, numParameters-{}, likelihood-{}'.format(numStates, numParameters, calculateAIC(numParameters, likelihood))

    numStates = 20
    numItems = 1621
    numItems = 5264  # Netflix
    likelihood = -3746421
    numParameters = ihmm(numStates, numItems)
    print 'IHMM: numStates-{}, numParameters-{}, likelihood-{}'.format(numStates, numParameters, calculateAIC(numParameters, likelihood))

    numStates = 30
    numItems = 1621
    numItems = 5264  # Netflix
    likelihood = -3677849
    numParameters = ihmm(numStates, numItems)
    print 'IHMM: numStates-{}, numParameters-{}, likelihood-{}'.format(numStates, numParameters, calculateAIC(numParameters, likelihood))

    numStates = 40
    numItems = 1621
    likelihood = -1230340
    numParameters = ihmm(numStates, numItems)
    print 'IHMM: numStates-{}, numParameters-{},likelihood-{}'.format(numStates, numParameters, calculateAIC(numParameters, likelihood))


    print hmm(10, numItems)
    print hmm(20, numItems)
    print hmm(30, numItems)
    print hmm(10, numItems) - hmm(30, numItems)