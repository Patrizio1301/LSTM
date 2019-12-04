import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def input_data(stock_data):
    stock_data["average"] = (stock_data['High'] + stock_data["Low"])/2
    input_feature = stock_data.iloc[:, [2, 6]].values
    input_data = input_feature
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0, 1))
    input_data[:, 0:2] = sc.fit_transform(input_feature[:, :])
    lookback = 50

    test_size = int(.3 * len(stock_data))
    X = []
    y = []
    for i in range(len(stock_data)-lookback-1):
        t = []
        for j in range(0,lookback):

            t.append(input_data[[(i+j)], :])
        X.append(t)
        y.append(input_data[i + lookback, 1])

    X, y = np.array(X), np.array(y)
    X_test = X[:test_size+lookback]
    Y_test = y[:test_size+lookback]
    X = X.reshape(X.shape[0], lookback, 2)
    X_test = X_test.reshape(X_test.shape[0], lookback, 2)
    return input_data, X, y, X_test, Y_test, lookback, test_size


