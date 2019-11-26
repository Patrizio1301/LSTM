from __future__ import print_function
import functools
import tensorflow as tf

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


def regularization(function):
    @functools.wraps(function)
    def wrapper(self, *args):
        if self.regularization is None:
            return function(self, *args)
        return tf.add(function(self, *args), self.regularization.regularization(self.weights))
    return wrapper


def distance(function):
    @functools.wraps(function)
    def wrapper(self, *args):
        if self.distance is None:
            return function(self, *args)
        return tf.add(function(self, *args), self.regularization.regularization(self.weights))
    return wrapper
