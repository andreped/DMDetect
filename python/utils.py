import tensorflow as tf
import numpy as np


def macro_accuracy(y_true, y_pred):
	y_pred_ = tf.argmax(y_pred, axis=1)
	y_true_ = tf.argmax(y_true, axis=1)
	nb_classes = 2  # assuming two classes
	S = 0
	for i in range(nb_classes):
		y_pred_tmp = tf.boolean_mask(y_pred_, y_true_ == i)
		y_true_tmp = tf.boolean_mask(y_true_, y_true_ == i)
		tmp = tf.cast(y_pred_tmp == y_true_tmp, dtype=tf.float32)
		S += tf.reduce_mean(tmp)
	S /= nb_classes  # to produce the macro mean
	return S


# @TODO: perhaps just use .ravel() instead?
def flatten_(tmp):
	out = []
	for t in tmp:
		for t2 in t:
			out.append(t2)
	out = np.array(out)
	return out


def minmaxscale(tmp, scale_=1):
	if np.count_nonzero(tmp) > 0:
		tmp = tmp - np.amin(tmp)
		tmp = tmp / np.amax(tmp)
		tmp *= scale_
	return tmp


# @TODO: Something wrong with this
def random_shift(x, aug, p=0.5):
    if  tf.random.uniform([]) < p:
        shapes = tf.shape(x)
        v1 = tf.cast(aug[0] * shapes[1], tf.int32)
        v2 = tf.cast(aug[1] * shapes[2], tf.int32)
        ha = tf.random.uniform([], minval=-5.5, maxval=5.5)
        wa = tf.cast(tf.random.uniform([], minval=-v2, maxval=v2), tf.int32)
        x = tfa.image.translate(x, [ha, wa], interpolation='nearest', fill_mode='constant', fill_value=0.)
    return x