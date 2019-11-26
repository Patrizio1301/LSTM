from __future__ import print_function
import tensorflow as tf
import numpy as np
from src.main.utils.decorators import lazy_property
from src.main.dataset.datasets import Datasets
from src.main.config.config import Config


class Parameters:
    """Holds model config

      The Parameters class is used to store weights and bias values. If this initial
      values will not be levegered, the Model class sets them randomly.
      """

    def __init__(self, weights=None, bias=None):
        self.weights = weights
        self.bias = bias


class Model:
    """Model Class for supervised machine learning models.

    The Model class is the main structure almost all
    supervised models in Machine learning should follow.


    Attributes
    ----------
    dataset : str
        a formatted string to print out what the animal says
    config : Config
        the name of the animal
    parameters : Parameters
        the sound that the animal makes

    Methods
    -------
    weights : tf.Variable
        variable for weight config
    bias : tf.Variable
        variable for bias config
    prediction : tf.Tensor
        Model prediction hypothesis
    cost : tf.Tensor
        Model cost function
    optimization : tf.Tensor
        Model optimization method
    error : tf.Tensor
        Model error calculation
    train: str
        Model training execution
    test: str
        Model testing execution
    save:
        Save training and testing results in json file.


    """

    def __init__(self, dataset: Datasets, config: Config, parameters: Parameters = Parameters()):
        self.config = config
        self.parameters = parameters
        self.dataset = dataset

    @lazy_property
    def weights(self):
        """Initial weights.

          The Parameters class is used to store weight values. If this initial
          values will not be levegered, the Model class sets them randomly.
          """
        if self.parameters.weights is not None:
            return tf.Variable(self.parameters.weights,
                               name="weights",
                               dtype=tf.float32)
        else:
            random_weights = np.random.rand(self.config.feature_num, 1)
            return tf.Variable(random_weights,
                               name="weights",
                               dtype=tf.float32)

    @lazy_property
    def bias(self):
        """Initial bias.

          The Parameters class is used to store bias values. If this initial
          values will not be levegered, the Model class sets them randomly.
          """
        if self.parameters.bias is not None:
            return tf.Variable(self.parameters.bias,
                               name="weights",
                               dtype=tf.float32)
        else:
            random_bias = np.random.rand(1, 1)
            return tf.Variable(random_bias,
                               name="weights",
                               dtype=tf.float32)

    @lazy_property
    def prediction(self):
        """Model prediction hypothesis.

           This method mainly depends on the choosen weights and bias.
           Cost and optimization methods will help to improve weights and bias.
           """
        raise NotImplementedError("Each Model needs a prediction method.")

    @lazy_property
    def cost(self):
        """Model cost function.

           This methods evaluates quantitaively how much the prediction
           is away from the "historical" truth.
           """
        raise NotImplementedError("Each Model needs a cost method.")

    @lazy_property
    def optimization(self):
        """Model optimization method.

           This method implements a manner to optimize weights and bias by
           minimizing the cost function.
           """
        raise NotImplementedError("Each Model needs a optimization method.")

    @lazy_property
    def error(self):
        """Validation function.

           This function evaluates the final output. In regression analysis this
           will be mainly the cost function, but in classification analysis this
           may be the accuracy.
           """
        raise NotImplementedError("Each Model needs an error method.")

    def training(self, session):
        """ Model training execution.

           This function evalues the final output. In regression analysis this
           will be mainly the cost function, but in classification analysis this
           may be the accuracy.
           """
        for epoch in range(self.config.epoche):
            session.run([self.dataset.training_data_op.initializer])
            while True:
                try:
                    print("uhuhu", session.run(self.dataset.training_data_next.feature))
                    _, error = session.run([self.optimization, self.error])
                except tf.errors.OutOfRangeError:
                    break
            print('Epoch {:2d} error {}'.format(epoch + 1, error))

    def test(self):
        """Model test execution.

        """
        # raise NotImplementedError("Each Model needs a test method.")

    def save(self,
             link,
             sess,
             global_step=None,
             latest_filename=None,
             meta_graph_suffix="meta",
             write_meta_graph=True,
             write_state=True,
             strip_default_attrs=False):
        """Saves Model config

        This method runs the ops added by the constructor for saving variables.
        It requires a session in which the graph was launched.  The variables to
        save must also have been initialized.

        The method returns the path prefix of the newly created checkpoint files.
        This string can be passed directly to a call to `restore()`.
        """
        save=tf.train.Saver()
        save.save(
            sess=sess,
            save_path=link,
            global_step=global_step,
            latest_filename=latest_filename,
            meta_graph_suffix=meta_graph_suffix,
            write_meta_graph=write_meta_graph,
            write_state=write_state,
            strip_default_attrs=strip_default_attrs
        )
