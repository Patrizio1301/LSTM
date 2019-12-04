from unittest import TestCase
import tensorflow as tf
import pandas as pd
import numpy as np
from src.main.dataset.datasets import Datasets

class TestDatasets(tf.test.TestCase):
    def setUp(self):
        """Set up all datasets in order to

        """
        feature_data= [[9, 10], [11, 15], [10, 14],  [10, 15],  [10, 16]]
        target_data = [[1], [0], [1], [0], [1]]
        feature_df=pd.DataFrame(feature_data, columns=["X_1", "X_2"])
        target_df=pd.DataFrame(target_data, columns=["Y"])
        self.feature_tensor=tf.convert_to_tensor(feature_df)
        self.target_tensor=tf.convert_to_tensor(target_df)
        self.batch_size=30
        self.training_size=0.4
        self.dataset=Datasets(self.feature_tensor,
                              self.target_tensor,
                              self.batch_size,
                              self.training_size)
        self.sess=tf.Session()

    def tearDown(self):
        tf.reset_default_graph()

    def test_sample_size(self):
        """Sample size test

        Since the sample size is 3, this unit test checks that the class
        attribute self.dataset.sample_size concurs with this value.
        """
        self.assertEqual(self.dataset.sample_size, 5)

    def test_training_data(self):
        """Training dataset test

        This unit test checks that the class attribute training_data
        of the class self.dataset contains the expected data.
        """
        self.sess.run(self.dataset.training_data_op.initializer)
        assert(self.sess.run(self.dataset.training_data_next),
               (np.array([[ 9, 10],[11, 15]]), np.array([[1],[0]])))

    def test_test_data(self):
        """Test dataset test

        This unit test checks that the class attribute test_data
        of the class self.dataset contains the expected data.
        """
        test_data = self.dataset.test_data.batch(3).make_initializable_iterator()
        self.sess.run(test_data.initializer)
        assert (self.sess.run(test_data.initializer),
                (np.array([[10, 14],[10, 15],[10, 16]]), np.array([[1], [0], [1]])))
