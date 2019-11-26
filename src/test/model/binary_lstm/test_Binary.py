import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.main.dataset.datasets import Datasets
from src.main.model.binary_lstm.binary import BinaryLSTM
from src.main.config.config import Config
from src.main.model.model import Parameters

num_steps = 5 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 10
num_classes = 2
state_size = 4
learning_rate = 0.1


def gen_data(size=100):
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    for i in range(size):
        threshold = 0.5
        if X[i-3] == 1:
            threshold += 0.5
        if X[i-8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)


def gen_batch(x, y, batch_size, num_steps):
    raw_x = x
    raw_y = y
    print(raw_x)
    print(raw_y)
    data_length = len(raw_x)

    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
    # further divide batch partitions into num_steps for truncated backprop
    epoch_size = batch_partition_length // num_steps

    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)


def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps)


x, y = gen_data()
a = gen_batch(x, y, batch_size, num_steps)

# data = tf.data.Dataset.from_tensor_slices((x, y))
#
# # sliding window batch
# data = data.window(10)
#
# data = data.interleave(lambda *window: tf.data.Dataset.zip(tuple([w.batch(batch_size) for w in window])),
#                         cycle_length=10, block_length=10, num_parallel_calls=4)
# # # data = data.apply(sliding.sliding_window_batch(window_size=window_size, window_shift=window_shift))
# # data = data.shuffle(1000, reshuffle_each_iteration=False)
# data = data.batch(5)
#
# # iter = dataset.make_initializable_iterator()
# init_op = data.make_initializable_iterator()
# print(init_op)
# el = init_op.get_next()
# print(el)
#
# NR_EPOCHS = 2
# with tf.Session() as sess:
#     for e in range (NR_EPOCHS):
#         print("\nepoch: ", e, "\n")
#         sess.run(init_op.initializer)
#         print("1  ", sess.run(el))
#         print("2  ", sess.run(el))



datos = Datasets(features=x, target=y, batch_size=5, window_size=10, training_size=1)

NR_EPOCHS = 2
with tf.Session() as sess:
    for e in range (NR_EPOCHS):
        print("\nepoch: ", e, "\n")
        sess.run(datos.training_data_op.initializer)
        print("1  ", sess.run(datos.training_data_next.feature))
        print("2  ", sess.run(datos.training_data_next.feature))

configuration = Config(feature_num=1,
                       batch_size=5,
                       epoche=10,
                       learning_rate=0.05,
                       min_learning_rate=0.001,
                       num_layers=1,
                       num_unrollings=1,
                       num_nodes=1,
                       dropout=0.1,
                       state_size=5)


modelo = BinaryLSTM(data_set=datos, config=configuration)

sess = tf.Session()
modelo.training(sess)
