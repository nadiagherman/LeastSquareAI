import csv
import os

import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from myLinearRegression import MyLinearMultivariateRegression


def loadData(fileName, inputVariabName1, inputVariabName2, outputVariabName):
    data = []
    dataNames = []
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                dataNames = row
            else:
                data.append(row)
            line_count += 1
    selectedVariable1 = dataNames.index(inputVariabName1)
    selectedVariable2 = dataNames.index(inputVariabName2)
    inputs = [[float(data[i][selectedVariable1]), float(data[i][selectedVariable2])] for i in range(len(data))]
    selectedOutput = dataNames.index(outputVariabName)
    outputs = [float(data[i][selectedOutput]) for i in range(len(data))]

    return inputs, outputs


def main():
    crtDir = os.getcwd()
    filePath = os.path.join(crtDir, 'world-happiness-report-2017.csv')

    inputs, outputs = loadData(filePath, 'Economy..GDP.per.Capita.', 'Freedom', 'Happiness.Score')
    # print('in:  ', inputs[:5])
    # print('out: ', outputs[:5])
    # np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = [i for i in indexes if not i in trainSample]
    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    # training step
    xx = [el for el in trainInputs]
    print(str(trainInputs))
    print(str(trainOutputs))

    #  with SKLEARN
    regressor = linear_model.LinearRegression()
    regressor.fit(xx, trainOutputs)
    w0, w1, w2 = regressor.intercept_, regressor.coef_[0], regressor.coef_[1]
    print('the learnt model with sklearn : f(x) = ', w0, ' + ', w1, ' * x', ' +', w2, ' * x^2')
    computedTestOutputs = [w0 + w1 * el[0] + w2 * el[1] for el in testInputs]
    # using sklearn
    errorSq = mean_squared_error(testOutputs, computedTestOutputs)
    print("prediction error (tool): ", errorSq)

    myRegressor = MyLinearMultivariateRegression()
    myRegressor.fit(xx, trainOutputs)
    mw0, mw1, mw2 = myRegressor.intercept_, myRegressor.coef_[0], myRegressor.coef_[1]
    print('the learnt model: f(x) = ', mw0, ' + ', mw1, ' * x', ' +', mw2, ' * x^2')
    myComputedTestOutputs = myRegressor.predict([x for x in testInputs])

    error = 0.0
    for t1, t2 in zip(myComputedTestOutputs, testOutputs):
        # using formula : error = ( y[i] - f(x[i]) ) ** 2 for i in range(len(myComputedTestOutputs))
        error += (t1 - t2) ** 2
    error = error / len(testOutputs)
    print('prediction error (manual): ', error)


main()
