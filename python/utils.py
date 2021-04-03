import tensorflow as tf
import numpy as np
import cv2


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


def IOU(y_true, y_pred, eps=1e-15, remove_bg=True):
	nb_classes = y_true.shape[-1]
	iou_ = 0
	for c in range(int(remove_bg), nb_classes):
		y_pred_curr = y_pred[..., c]
		y_true_curr = y_true[..., c]
		intersection = (y_true_curr * y_pred_curr).sum()
		union = y_true_curr.sum() + y_pred_curr.sum() - intersection
		iou_ += (intersection + eps) / (union + eps)
	iou_ /= (nb_classes - int(remove_bg))
	return iou_


def DSC(y_true, y_pred, smooth=1e-15, remove_bg=True):
	nb_classes = int(y_true.shape[-1])
	dice = 0
	for c in range(int(remove_bg), nb_classes):
		y_pred_curr = y_pred[..., c]
		y_true_curr = y_true[..., c]
		intersection1 = np.sum(y_pred_curr * y_true_curr)
		union1 = np.sum(y_pred_curr * y_pred_curr) + np.sum(y_true_curr * y_true_curr)
		dice += (2. * intersection1 + smooth) / (union1 + smooth)
	dice /= (nb_classes - int(remove_bg))
	return dice


def one_hot_fix(x, nb_classes):
	out = np.zeros(x.shape + (nb_classes,), dtype=np.int32)
	for c in range(nb_classes):
		out[..., c] = (x == c).astype(np.int32)
	return out


def argmax_keepdims(x, axis):
	output_shape = list(x.shape)
	output_shape[axis] = 1
	return np.argmax(x, axis=axis).reshape(output_shape)


def post_process(x, new_shape, orig_shape, resize=True, interpolation=cv2.INTER_NEAREST, threshold=None):
	x = x.astype(np.float32)
	x = x[:new_shape[0], :new_shape[1]]
	if resize:
		new = np.zeros(orig_shape + (x.shape[-1],), dtype=np.float32)
		for i in range(x.shape[-1]):
			new[..., i] = cv2.resize(x[..., i], orig_shape[::-1], interpolation=interpolation)
		if interpolation != cv2.INTER_NEAREST:
			if threshold is not None:
				new = (new > 0.5).astype(np.int32)
		return new
	else:
		return x