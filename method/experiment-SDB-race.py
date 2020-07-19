#!/usr/bin/env python3

from data import adult2

from errorfunctions import signedStatisticalParity, labelError, individualFairness
from utils import errorBars, arrayErrorBars, experimentCrossValidate
from margin import svmRBFMarginAnalyzer, svmLinearMarginAnalyzer, boostingMarginAnalyzer, lrSKLMarginAnalyzer


def lrLearner(train, protectedIndex, protectedValue):
    marginAnalyzer = lrSKLMarginAnalyzer(train, protectedIndex, protectedValue)
    shift = marginAnalyzer.optimalShift()
    print('best shift is: %r' % (shift,))
    return marginAnalyzer.conditionalShiftClassifier(shift)


def boostingLearner(train, protectedIndex, protectedValue):
    marginAnalyzer = boostingMarginAnalyzer(train, protectedIndex, protectedValue)
    shift = marginAnalyzer.optimalShift()
    print('best shift is: %r' % (shift,))
    return marginAnalyzer.conditionalShiftClassifier(shift)


def svmLearner(train, protectedIndex, protectedValue):
    marginAnalyzer = svmRBFMarginAnalyzer(train, protectedIndex, protectedValue)
    shift = marginAnalyzer.optimalShift()
    print('best shift is: %r' % (shift,))
    return marginAnalyzer.conditionalShiftClassifier(shift)


def svmLinearLearner(train, protectedIndex, protectedValue):
    marginAnalyzer = svmLinearMarginAnalyzer(train, protectedIndex, protectedValue)
    shift = marginAnalyzer.optimalShift()
    print('best shift is: %r' % (shift,))
    return marginAnalyzer.conditionalShiftClassifier(shift)


@arrayErrorBars(2)
def statistics(train, test, protectedIndex, protectedValue, learner):
    h = learner(train, protectedIndex, protectedValue)
    print("Computing error")
    error = labelError(test, h)
    print("Computing bias")
    bias = signedStatisticalParity(test, protectedIndex, protectedValue, h)
    print("Computing UBIF")
    ubif = individualFairness(train, learner, 0.2, passProtected=True)
    return error, bias, ubif


@errorBars(10)
def indFairnessStats(train, learner):
    print("Computing UBIF")
    ubif = individualFairness(train, learner, flipProportion=0.2, passProtected=True)
    print("UBIF:", ubif)
    return ubif


def runAll():
    print("Shifted Decision Boundary Relabeling")
    experiments = [(('SVM', svmLearner), adult2),
                   (('AdaBoost', boostingLearner), adult2),
                   (('LR', lrLearner), adult2),
                   ]

    for (learnerName, learner), dataset in experiments:
        print("%s %s" % (dataset.name, learnerName), flush=True)
        experimentCrossValidate(dataset, learner, 2, statistics)


if __name__ == '__main__':
    runAll()
