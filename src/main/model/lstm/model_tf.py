# Working example for my blog post at:
# https://danijar.github.io/structuring-your-tensorflow-models
import tensorflow as tf
from src.main.model.model import Config
import numpy as np
import matplotlib.pyplot as plt
from src.main.model.lstm.utils import define_scope


# Global config variables
num_steps = 5 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 200
num_classes = 2
state_size = 4
learning_rate = 0.05


class Model:

    def __init__(self, feature, target, init_state, weights, bias, config: Config):
        self.config = config
        self.feature = feature
        self.target = target
        self.init_state = init_state
        self.weights = weights
        self.bias = bias
        self.prediction
        self.optimization
        self.error
        self.cost
        self.cell_outputs
        self.logits
        self.final_state
        self.outputs

    # @share_variables
    # def init_state(self):
    #     return tf.zeros([self.config.batch_size, self.config.state_size])

    @define_scope
    def embeddings(self):
        embeddings = tf.get_variable('embedding_matrix', [self.config.num_classes, self.config.state_size])
        return tf.nn.embedding_lookup(embeddings, self.feature)

    @define_scope
    def cell_outputs(self):
        """Prediction hypothesis for this model.

        This method predicts the output with an LSTMCell.
        """
        cell1 = tf.nn.rnn_cell.LSTMCell(30, state_is_tuple=True)
        cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, input_keep_prob=self.config.dropout)
        cell2 = tf.nn.rnn_cell.LSTMCell(30, state_is_tuple=True)
        cell2 = tf.nn.rnn_cell.DropoutWrapper(cell2, input_keep_prob=self.config.dropout)
        cell3 = tf.nn.rnn_cell.LSTMCell(30, state_is_tuple=True)
        cell3 = tf.nn.rnn_cell.DropoutWrapper(cell3, input_keep_prob=self.config.dropout)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2, cell3], state_is_tuple=True)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.config.dropout)
        rnn_inputs = tf.one_hot(self.feature, self.config.num_classes)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, self.embeddings, initial_state=self.init_state)
        rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
        return [rnn_outputs, final_state]

    @define_scope
    def final_state(self):
        return self.cell_outputs[1]

    @define_scope
    def outputs(self):
        return self.cell_outputs[0]

    @define_scope
    def prediction(self):
        """Prediction hypothesis for this model.

        This method predicts the output with an LSTMCell.
        """
        prediction_resized = tf.reshape(self.outputs, [-1, self.config.state_size])
        return tf.matmul(prediction_resized, self.weights) + self.bias

    @define_scope
    def cost(self):
        """Cost function for this model.

        This method returns the formula sum((Yhat-Y)^2)/n, where X is just a batch of
        the feature data, Y is a batch of the target data, and n is the dimension.
        """
        losses = tf.pow(tf.subtract(self.target, logits=self.prediction), 2)
        return tf.reduce_mean(losses)

    @define_scope
    def optimization(self):
        """Gradient descent optimization.

        After each batch, this optimization method updates the weights and bias.
        """
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        return optimizer.minimize(self.cost)

    @define_scope
    def error(self):
        return self.cost