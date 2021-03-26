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