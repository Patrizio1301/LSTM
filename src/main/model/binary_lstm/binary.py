import tensorflow as tf
from src.main.model.model import Model, Parameters, Config
from src.main.utils.decorators import lazy_property
from src.main.dataset.datasets import Datasets


class BinaryLSTM(Model):
    def __init__(self, data_set: Datasets, config: Config, parameters: Parameters = None):
        Model.__init__(self, dataset=data_set, config=config, parameters=parameters)

    @lazy_property
    def init_state(self):
        return tf.zeros([self.config.batch_size, self.config.state_size])

    @lazy_property
    def cell_outputs(self):
        """Prediction hypotesis for this model.

        This method predicts the output with an LSTMCell.
        """
        cell = tf.keras.layers.LSTMCell(self.config.state_size)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, self.dataset.training_data_next.feature, initial_state=self.init_state)
        return rnn_outputs

    @lazy_property
    def logits(self):
        prediction_resized = tf.reshape(self.cell_outputs, [-1, self.config.state_size])
        logits2 = tf.matmul(prediction_resized, self.weights) + self.bias
        logits_resized = tf.reshape(logits2, [self.config.batch_size, self.config.num_steps, self.config.num_classes])
        return logits_resized

    @lazy_property
    def prediction(self):
        """Prediction hypotesis for this model.

        This method predicts the output with an LSTMCell.
        """
        return tf.nn.softmax(self.logits)

    @lazy_property
    def cost(self):
        """Cost function for this model.

        This method returns the formula sum((X*W+b-Y)^2)/n, where X is just a batch of
        the feature data, Y is a batch of the target data, and n is the dimension.
        """
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.dataset.training_data_next.target, logits=self.logits)
        return tf.reduce_mean(losses)

    @lazy_property
    def optimization(self):
        """Gradient descent optimization.

        After each batch, this optimization method updates the weights and bias.
        """
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        return self.cost
