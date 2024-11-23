import numpy as np
import tensorflow as tf


class R2LossNoWeights(tf.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_sum(tf.square(y_true - y_pred)) / tf.reduce_sum(tf.square(y_true))


def r2_metric(y_true, y_pred, weight):
    weighted_r2 = 1 - (np.sum(weight * (y_true - y_pred) ** 2) / np.sum(weight * y_true ** 2))
    return weighted_r2
