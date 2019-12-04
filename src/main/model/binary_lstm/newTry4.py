# Working example for my blog post at:
# https://danijar.github.io/structuring-your-tensorflow-models
import functools
import tensorflow as tf
from src.main.model.model import Config
import numpy as np
import matplotlib.pyplot as plt


# Global config variables
num_steps = 5 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 200
num_classes = 2
state_size = 4
learning_rate = 0.1


def gen_data(size=1000000):
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


# adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb/reader.py
def gen_batch(raw_data, batch_size, num_steps):
    raw_x, raw_y = raw_data
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


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if no arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

share_variables = lambda func: tf.make_template(
    func.__name__, func, create_scope_now_=True)


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


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

    # @share_variables
    # def init_state(self):
    #     return tf.zeros([self.config.batch_size, self.config.state_size])

    @define_scope
    def cell_outputs(self):
        """Prediction hypotesis for this model.

        This method predicts the output with an LSTMCell.
        """
        cell = tf.contrib.rnn.BasicRNNCell(self.config.state_size)
        rnn_inputs = tf.one_hot(self.feature, self.config.num_classes)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=self.init_state)
        return rnn_outputs

    @define_scope
    def final_state(self):
        """Prediction hypotesis for this model.

        This method predicts the output with an LSTMCell.
        """
        cell = tf.contrib.rnn.BasicRNNCell(self.config.state_size)
        rnn_inputs = tf.one_hot(self.feature, self.config.num_classes)

        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, rnn_inputs, initial_state=self.init_state)
        return final_state

    @define_scope
    def logits(self):
        prediction_resized = tf.reshape(self.cell_outputs, [-1, self.config.state_size])
        logits2 = tf.matmul(prediction_resized, self.weights) + self.bias
        logits_resized = tf.reshape(logits2, [self.config.batch_size, self.config.num_steps, self.config.num_classes])
        return logits_resized

    @define_scope
    def prediction(self):
        """Prediction hypotesis for this model.

        This method predicts the output with an LSTMCell.
        """
        return tf.nn.softmax(self.logits)

    @define_scope
    def cost(self):
        """Cost function for this model.

        This method returns the formula sum((X*W+b-Y)^2)/n, where X is just a batch of
        the feature data, Y is a batch of the target data, and n is the dimension.
        """
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=self.logits)
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

    # def define_graph(self):
    #     tf.compat.v1.reset_default_graph()
    #     feature = tf.compat.v1.placeholder(tf.int32, [batch_size, num_steps])
    #     target = tf.compat.v1.placeholder(tf.int32, [batch_size, num_steps])
    #     init_state = tf.compat.v1.placeholder(tf.int32, [batch_size, num_steps])
    #     prediction = self.prediction(feature, init_state)
    #     loss = self.cost(target, feature, init_state)
    #     final_state = self.final_state(feature, init_state)
    #     optimization = self.optimization(target, feature, init_state)
    #     return AttrDict(locals())


def main(config, num_epochs, num_steps, state_size=4, verbose=True):
    feature = tf.compat.v1.placeholder(tf.int32, [batch_size, num_steps])
    target = tf.compat.v1.placeholder(tf.int32, [batch_size, num_steps])
    init_state = tf.zeros([batch_size, state_size])
    with tf.variable_scope('softmax'):
        weights = tf.get_variable('W', [state_size, num_classes])
        bias = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))
    print(weights, bias)
    model = Model(feature=feature,
                  target=target,
                  init_state=init_state,
                  weights=weights,
                  bias=bias,
                  config=config)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    training_losses = []
    for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
        training_loss = 0
        training_state = np.zeros((batch_size, state_size))
        if verbose:
            print("\nEPOCH", idx)
        for step, (X, Y) in enumerate(epoch):
            training_loss_, training_state, _ = \
                sess.run([model.cost,
                          model.final_state,
                          model.optimization], feed_dict={feature: X, target: Y, init_state: training_state})
            training_loss += training_loss_
            if step % 100 == 0 and step > 0:
                if verbose:
                    print("Average loss at step", step,
                              "for last 250 steps:", training_loss/100)
                training_losses.append(training_loss/100)
                training_loss = 0

    return training_losses


if __name__ == '__main__':
    configuration = Config(feature_num=1,
                           batch_size=batch_size,
                           epoche=20,
                           learning_rate=learning_rate,
                           min_learning_rate=learning_rate,
                           num_layers=1,
                           num_unrollings=1,
                           num_classes=num_classes,
                           num_nodes=1,
                           dropout=0.1,
                           state_size=state_size,
                           num_steps=num_steps)
    training_losses = main(config=configuration, num_epochs=1, num_steps=num_steps)
    plt.plot(training_losses)
    plt.show()