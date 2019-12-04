import tensorflow as tf
from src.main.model.model import Model, Parameters, Config
from src.main.utils.decorators import lazy_property


def keras_model(X):
    from keras import Sequential
    from keras.layers import Dense, LSTM
    model = Sequential()
    model.add(LSTM(units=30, return_sequences=True, input_shape=(X.shape[1], 2)))
    model.add(LSTM(units=30, return_sequences=True))
    model.add(LSTM(units=30))
    model.add(Dense(units=1))
    return model




