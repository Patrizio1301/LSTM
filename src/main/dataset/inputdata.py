import tensorflow as tf
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from .datasets import Datasets


@dataclass(frozen=True)
class Input:
    x: object
    y: object

def regression_data(TRUE_W, TRUE_b: float, NUM_EXAMPLES: int, batch_size: int):
    x = tf.random_normal(shape=[NUM_EXAMPLES, len(TRUE_W)])
    noise = tf.random_normal(shape=[NUM_EXAMPLES, 1])*10
    y = tf.matmul(x,tf.constant(TRUE_W, dtype=tf.float32)) + TRUE_b + noise
    return Datasets(x, y, batch_size)

def classification_data(batch_size,**args):
    data=datasets.make_classification(**args)
    return Datasets(
        tf.convert_to_tensor(data[0], dtype=tf.float32),
        tf.reshape(tf.convert_to_tensor(data[1], dtype=tf.float32), [-1, 1]),
        batch_size
    )


if __name__ == '__main__':
    TRUE_W = [[10.0], [2]]
    TRUE_b = 5.0
    NUM_EXAMPLES = 100000000
    dataSet = regression_data(TRUE_W, TRUE_b, NUM_EXAMPLES)
    sess = tf.Session()
    print(sess.run([dataSet.output]))


