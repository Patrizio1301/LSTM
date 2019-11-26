import tensorflow as tf
import numpy as np
from src.main.utils.decorators import lazy_property


class Config:
    """Holds model hyperparams and data information.

      The config class is used to store various hyperparameters and dataset
      information config. Model objects are passed a Config() object at
      instantiation.

    Attributes:
    ------------
    Data-related-configs:
    ---------------------
    feature_num:
        Number of feature elements

    training-configs:
    ---------------------
    epoche:
        Number of epoches which should be executed
    batch_size:
        Batch size is how many data samples you consider in a single time step.
        The larger the better, because more visibility of data you have at a given time.
    learning_rate:



    model-config:
    ---------------------
    n_layers:
        number of layers
    num_unrollings:
         denotes how many continuous time steps you consider for a single
         optimization step. The larger the better.
    num_nodes:
        number of hidden neurons in each cell.
    dropout:

      """

    def __init__(self,
                 feature_num: int,
                 batch_size: int,
                 epoche: int,
                 learning_rate: float,
                 min_learning_rate: float,
                 num_layers: int,
                 num_unrollings: int,
                 num_nodes: list,
                 dropout: float,
                 state_size: int,
                 num_classes,
                 num_steps):
        self.epoche = epoche
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.batch_size = batch_size
        self.feature_num = feature_num
        self.num_layers = num_layers
        self.num_unrollings = num_unrollings
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.state_size = state_size
        self.num_classes = num_classes
        self.num_steps = num_steps

    def dynamical_learning_rate(self, global_step):
        learning_rate = tf.maximum(
            tf.train.exponential_decay(
                self.learning_rate,
                global_step,
                decay_steps=1,
                decay_rate=0.5,
                staircase=True),
            self.min_learning_rate)
        return learning_rate
