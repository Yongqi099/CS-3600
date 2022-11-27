from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData
from NeuralNet import buildNeuralNet
from math import pow, sqrt

import logging
logging.basicConfig(level=logging.DEBUG, filename="Question 6.log", filemode="w")

def maxAccuracy(argList):
    return max(argList)
    maxAcc = float("-inf")
    for arg in argList:
        maxAcc = max(arg, maxAcc)
    return maxAcc

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

penData = buildExamplesFromPenData()
def testPenData(hiddenLayers):
    return buildNeuralNet(penData, maxItr = 200, hiddenLayerList = hiddenLayers)

carData = buildExamplesFromCarData()
def testCarData(hiddenLayers):
    return buildNeuralNet(carData, maxItr = 200,hiddenLayerList = hiddenLayers)

def getAll(argList):
    maxAcc = maxAccuracy(argList)
    avg = average(argList)
    st = stDeviation(argList)
    return ('Max: %f\nAvg: %f\nSt:%f\n' % (maxAcc, avg, st))

def testAll(num_perceptrons, iteration):
    pen_result = []
    car_result = []
    for i in range(5):
        # pen_result.append(testPenData([num_perceptrons]))
        car_result.append(testCarData([num_perceptrons]))

    # logging.info('Pen Iteration #%d - %d' % (iteration, num_perceptrons))
    # getAll(pen_result)
    logging.info('\nCar Iteration #%d - %d\n%s' % (iteration, num_perceptrons, getAll(car_result)))


for num_perceptrons in range(0, 45, 5):
    testAll(num_perceptrons, num_perceptrons/5)