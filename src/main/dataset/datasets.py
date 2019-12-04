import tensorflow as tf
from tensorflow import Tensor
from src.main.utils.decorators import lazy_property


class Datasets(object):
    """Input pipeline using tf.data.dataset API

    DataSet input pipeline which will be used as an input for a model.
    This class separates your input into training and test data and
    cluster your data into batches with size batch_size.

    Attributes
    ----------
    features : tf.Tensor
        features data. This will define the feature space X.
    target : tf.Tensor
        target data. This defines the target space Y.
    batch_size : int
        Batch size for batch/mini-batch optimization. If the
        batch_size is equal to None (by default), the optimization
        will be batch optimization. If batch_size is equal to -1,
        this will be gradient optimization. If batch_size is a
        natural number, mini batch optimization will be applied.
    training_size: float
        Percentage of desired training set. For example, if this
        parameter is initialized with the value 0.8, 80 percentage
        of the input data will be used as the training data.

    Methods
    -------
    sample_size : int
        DataSet sample size. If the given feature and target
        inputs do not concur in their sample size, this method
        will rise an ValueError and this object will be useless.
    dataSet : tf.Tensor
        DataSet. The dataSet is defined as a zipped dataSet with
        the structure (feature, target) for each sample.
    training_data : tf.Tensor
        Training dataSet.
    training_data_op : Operator
        Iterator initializer.
    training_data_next : tf.Tensor
        Next training data batch.
    test_data : tf.Tensor
        Testing dataSet
    test_data_op : Operator
        Iterator initializer.
    test_data_next : tf.Tensor
        Next test data batch.
    """

    def __init__(self, features: Tensor, target: Tensor,
                 features_test: Tensor = None, target_test: Tensor = None,
                 batch_size=None, window_size=None, training_size=0.8):
        self.feature_data = features
        self.target_data = target
        self.feature_data_test = features_test
        self.target_data_test = target_test
        self.batch_size = batch_size
        self.training_size = training_size
        self.window_size = window_size

    class DatasetXY(object):
        """Class with two properties: feature and target
        """
        def __init__(self, df):
            self.df = df

        @lazy_property
        def feature(self):
            return self.df[0]

        @lazy_property
        def target(self):
            return self.df[1]

    @staticmethod
    def __batch_dataset__(batch_size, dataset):
        """Combines consecutive elements of this dataset into batches.

        The components of the resulting element will have an additional outer
        dimension, which will be `batch_size` (or `N % batch_size` for the last
        element if `batch_size` does not divide the number of input elements `N`
        evenly and `drop_remainder` is `False`). If your program depends on the
        batches having the same outer dimension, you should set the `drop_remainder`
        argument to `True` to prevent the smaller batch from being produced.

        Args:
          batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
            consecutive elements of this dataset to combine in a single batch.
          drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
            whether the last batch should be dropped in the case it has fewer than
            `batch_size` elements; the default behavior is not to drop the smaller
            batch.

        Returns:
          Dataset: A `Dataset`.
        """
        if batch_size is not None:
            return dataset.batch(batch_size)
        return dataset

    @staticmethod
    def __window_dataset__(window_size, dataset):
        """Combines (nests of) input elements into a dataset of (nests of) windows
           in case window_size is not None.

        A "window" is a finite dataset of flat elements of size `size` (or possibly
        fewer if there are not enough input elements to fill the window and
        `drop_remainder` evaluates to false).

        The `stride` argument determines the stride of the input elements, and the
        `shift` argument determines the shift of the window.

        For example, letting {...} to represent a Dataset:

        - `tf.data.Dataset.range(7).window(2)` produces
          `{{0, 1}, {2, 3}, {4, 5}, {6}}`
        - `tf.data.Dataset.range(7).window(3, 2, 1, True)` produces
          `{{0, 1, 2}, {2, 3, 4}, {4, 5, 6}}`
        - `tf.data.Dataset.range(7).window(3, 1, 2, True)` produces
          `{{0, 2, 4}, {1, 3, 5}, {2, 4, 6}}`

        Note that when the `window` transformation is applied to a dataset of
        nested elements, it produces a dataset of nested windows.

        For example:

        - `tf.data.Dataset.from_tensor_slices((range(4), range(4))).window(2)`
          produces `{({0, 1}, {0, 1}), ({2, 3}, {2, 3})}`
        - `tf.data.Dataset.from_tensor_slices({"a": range(4)}).window(2)`
          produces `{{"a": {0, 1}}, {"a": {2, 3}}}`

        Args:
          size: A `tf.int64` scalar `tf.Tensor`, representing the number of elements
            of the input dataset to combine into a window.
          shift: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
            forward shift of the sliding window in each iteration. Defaults to
            `size`.
          stride: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
            stride of the input elements in the sliding window.
          drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
            whether a window should be dropped in case its size is smaller than
            `window_size`.

        Returns:
          Dataset: A `Dataset` of (nests of) windows -- a finite datasets of flat
            elements created from the (nests of) input elements.

        """
        if window_size is not None:
            data = dataset.window(window_size)
            func = lambda *window: tf.data.Dataset.zip(tuple([w.batch(window_size) for w in window]))
            data = data.interleave(func,
                                   cycle_length=10,
                                   block_length=10,
                                   num_parallel_calls=4)
            return data
        return dataset

    @lazy_property
    def sample_size(self):
        """Sample size

        Sample size of dataset. Since features and target data are
        independent class arguments, their size does not necessarily
        match. In the case they do not, a ValueError exception is
        raised.

        Returns:
          int: integer of sample size.
        """
        feature_size = self.feature_data.shape[0]

        target_size = self.target_data.shape[0]
        if feature_size != target_size:
            raise ValueError("Feature sample size {} and target sample size "
                             "{} must be identically!".format(feature_size, target_size))
        else:
            return int(feature_size)

    @lazy_property
    def dataset(self):
        """Creates a `Dataset` by zipping together feature and target set

        This method has similar semantics to the built-in `zip()` function
        in Python, with the main difference being that the `datasets`
        argument can be an arbitrary nested structure of `Dataset` objects.
        For example:

        ```python
        feature = Dataset.range(1, 4)  # ==> [ 1, 2, 3 ]
        target = Dataset.range(4, 7)  # ==> [ 4, 5, 6 ]

        # The nested structure of the `datasets` argument determines the
        # structure of elements in the resulting dataset.
        Datasets.dataset  # ==> [ (1, 4), (2, 5), (3, 6) ]

        Returns:
            Dataset: A `Dataset`.
        """

        feature = tf.data.Dataset.from_tensor_slices(self.feature_data)
        target = tf.data.Dataset.from_tensor_slices(self.target_data)
        return tf.data.Dataset.zip((feature, target))

    @lazy_property
    def training_sample_size(self):
        """training sample size

        Size equal to the integer part of training size
        (percentage of dataset) multiplied by dataset sample size.

        Returns:
            Integer: A `Dataset`.
        """
        return int(self.training_size * self.sample_size)

    @lazy_property
    def training_data(self):
        """training dataset

        Training dataset with size equal to ´training_sample_size´.

        Returns:
          Dataset: A `Dataset`.
        """
        dataset = self.__window_dataset__(self.window_size, self.dataset)
        if self.feature_data_test is not None \
                and self.target_data_test is not None:
            return self.__batch_dataset__(self.batch_size,
                                          dataset)
        else:
            return self.__batch_dataset__(self.batch_size,
                                          dataset.take(self.training_sample_size))

    @lazy_property
    def training_data_op(self):
        """Iterator initializer of training data

         Returns:
            An `Iterator` over the elements of this dataset.
        """
        return self.training_data.make_initializable_iterator()

    @lazy_property
    def training_data_next(self):
        """Obtain next batch of training data

        """
        return self.DatasetXY(self.training_data_op.get_next())

    @lazy_property
    def test_data(self):
        """Dataset for testing

        Testing dataset with size equal to sample size minus the
        training dataset size.
        """
        if self.feature_data_test is not None and self.target_data_test is not None:
            feature = tf.data.Dataset.from_tensor_slices(self.feature_data_test)
            target = tf.data.Dataset.from_tensor_slices(self.target_data_test)
            return self.__batch_dataset__(1, tf.data.Dataset.zip((feature, target)))
        else:
            return self.__batch_dataset__(1, self.dataset.skip(self.training_sample_size))

    @lazy_property
    def test_data_op(self):
        """Iterator initializer of test data

        """
        return self.test_data.make_initializable_iterator()

    @lazy_property
    def test_data_next(self):
        """Obtain next batch of test data

        """
        return self.DatasetXY(self.test_data_op.get_next())