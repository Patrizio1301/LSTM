import pandas as pd
from src.main.model.lstm.model_weird import keras_model
from src.main.model.lstm.training import training
from src.main.model.lstm.data import input_data
import matplotlib.pyplot as plt
import tensorflow as tf


def main(path):
    stock_data = pd.read_csv(path)
    input, x, y, x_test, y_test, lookback, test_size = input_data(stock_data)
    print(x)
    model = keras_model(x)
    model = training(model, x, y, epoche=15)
    predicted_value = model.predict(x_test)
    print(lookback, test_size)

    plt.plot(predicted_value, color='red')
    plt.plot(input[lookback:test_size+(2*lookback), 1], color='green')
    plt.title("Opening price of stocks sold")
    plt.xlabel("Time (latest-> oldest)")
    plt.ylabel("Stock Opening Price")
    plt.show()

    y_test = tf.cast(y_test, tf.float64)
    predicted_value = tf.reshape(tf.cast(predicted_value, tf.float64), -1)
    print(tf.Session().run(R_squared(y_test, predicted_value)))


def R_squared(y, y_pred):
    print(tf.Session().run(y), tf.Session().run(y_pred))
    residual = tf.reduce_sum(tf.square(tf.subtract(y, y_pred)))
    total = tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y))))
    r2 = tf.subtract(tf.cast(1.0, tf.float64), tf.div(residual, total))
    return tf.subtract(y, y_pred)


if __name__ == '__main__':
    main('Google.csv')
