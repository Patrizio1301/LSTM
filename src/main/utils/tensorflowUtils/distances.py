from tensorflow import Tensor
import tensorflow as tf

class distances:

    @staticmethod
    def pointToMatrix(point: Tensor, matrix: Tensor):
        expanded_matrix = tf.cast(matrix, tf.float64)
        expanded_point = tf.cast(point, tf.float64)
        dist = tf.reduce_sum(tf.square(tf.subtract(expanded_matrix, expanded_point)), 1)
        return dist

    @staticmethod
    def matrixToMatrix(matrix1: Tensor, matrix2: Tensor):
        expanded_matrix1 = tf.cast(matrix1, tf.float64)
        expanded_matrix2 = tf.cast(matrix2, tf.float64)
        dist = tf.reduce_sum(tf.square(tf.subtract(expanded_matrix1, expanded_matrix2)), 2)
        return dist
