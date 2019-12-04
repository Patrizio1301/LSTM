import tensorflow as tf
from src.main.utils.decorators import lazy_property


class Dataset(object):
    def __init__(self, features, target, batch_size):
        self.feature_data=features
        self.target_data=target
        self.batch_size=batch_size
        self.features
        self.target
        self.features_op
        self.target_op
        self.features_batch
        self.target_batch

    @lazy_property
    def features(self):
        dataset=tf.data.Dataset.from_tensor_slices(self.feature_data).batch(self.batch_size)
        return dataset.make_initializable_iterator()

    @lazy_property
    def features_op(self):
        return self.features.initializer

    @lazy_property
    def features_batch(self):
        return self.features.get_next()

    @lazy_property
    def target(self):
        dataset=tf.data.Dataset.from_tensor_slices(self.target_data).batch(self.batch_size)
        return dataset.make_initializable_iterator()

    @lazy_property
    def target_op(self):
        return self.target.initializer

    @lazy_property
    def target_batch(self):
        return self.target.get_next()
