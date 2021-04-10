import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import disk, remove_small_holes, remove_small_objects, label


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
		intersection = (y_pred_curr * y_true_curr).sum()
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


def post_process_mammary_gland(pred, all_class_names):
	# orig classes
	orig = pred.copy()

	# fill holes in mammary gland class
	mamma_class = pred[..., np.argmax(all_class_names == "mammary_gland")]
	pred[..., np.argmax(all_class_names == "mammary_gland")] = remove_small_holes(mamma_class.astype(np.bool), area_threshold=int(np.round(np.prod(mamma_class.shape) / 1e3))).astype(np.float32)
	
	#'''
	# need to update mammary gland class by all other classes to avoid missing actual true positives
	tmp1 = pred[..., np.argmax(all_class_names == "mammary_gland")]
	for c in all_class_names:
		if c == "background":
			continue
		if c != "mammary_gland":
			tmp2 = pred[..., np.argmax(all_class_names == c)]
			tmp1[tmp2 == 1] = 0
	pred[..., np.argmax(all_class_names == "mammary_gland")] = tmp1

	# only keep largest mammary gland connected component
	labels = label(pred[..., np.argmax(all_class_names == "mammary_gland")])
	best = 0
	largest_area = 0
	for l in np.unique(labels):
		if l == 0:
			continue
		area = np.sum(labels == l)
		if area > largest_area:
			largest_area = area
			best = l
	pred[..., np.argmax(all_class_names == "mammary_gland")] = (labels == best).astype(np.float32)

	# fix background class
	tmp1 = pred[..., np.argmax(all_class_names == "background")]
	for c in all_class_names:
		if c != "background":
			tmp2 = pred[..., np.argmax(all_class_names == c)]
			tmp1[tmp2 == 1] = 0
	pred[..., np.argmax(all_class_names == "background")] = tmp1
	#'''

	# keep original cancer pred
	pred[..., np.argmax(all_class_names == "cancer")] = orig[..., np.argmax(all_class_names == "cancer")]  # .copy() TODO: Why does this .copy() change the output?

	return pred


def random_jet_colormap(cmap="jet", nb=256):
	colors = plt.cm.get_cmap(cmap, nb)  # choose which colormap to use
	tmp = np.linspace(0, 1, nb)
	np.random.shuffle(tmp)  # shuffle colors
	newcolors = colors(tmp)
	newcolors[0, :] = (0, 0, 0, 1)  # set first color to black
	return plt.cm.colors.ListedColormap(newcolors)


def make_subplots(x, y, pred, conf, img_size, all_class_names, some_cmap):

		nb_classes = y.shape[-1]
		'''
		fig1, ax1 = plt.subplots(1, 3)
		ax1[0].imshow(x_orig, cmap="gray")
		ax1[1].imshow(np.argmax(pred_orig, axis=-1), cmap="jet", vmin=0, vmax=nb_classes-1)
		ax1[2].imshow(np.argmax(y_orig, axis=-1), cmap="jet", vmin=0, vmax=nb_classes-1)
		plt.show()
		'''

		fig, ax = plt.subplots(4, nb_classes)
		ax[0, 0].imshow(x, cmap="gray", interpolation='none')
		ax[0, 1].imshow(np.argmax(pred, axis=-1), cmap=some_cmap, vmin=0, vmax=nb_classes-1, interpolation='none')
		ax[0, 2].imshow(np.argmax(y, axis=-1), cmap=some_cmap, vmin=0, vmax=nb_classes-1, interpolation='none')

		for i in range(nb_classes):
			ax[1, i].imshow(conf[..., i], cmap="gray", vmin=0, vmax=1, interpolation='none')
			ax[2, i].imshow(pred[..., i], cmap="gray", vmin=0, vmax=1, interpolation='none')
			ax[3, i].imshow(y[..., i], cmap="gray", vmin=0, vmax=1, interpolation='none')

		for i in range(nb_classes):
			for j in range(4):
				ax[j, i].axis("off")

		for i, cname in enumerate(all_class_names):
			ax[-1, i].text(int(pred.shape[1] * 0.5), img_size + int(img_size * 0.1), cname, color="g", verticalalignment='center', horizontalalignment='center')

		ax[0, 0].set_title('Img', color='c', rotation='vertical', x=-0.1, y=0.4)
		ax[1, 0].set_title('Conf', color='c', rotation='vertical', x=-0.1, y=0.4)
		ax[2, 0].set_title('Pred', color='c', rotation='vertical', x=-0.1, y=0.4)
		ax[3, 0].set_title('GT', color='c', rotation='vertical', x=-0.1, y=0.4)
		ax[0, 1].set_title('MC Pred', color='orange')
		ax[0, 2].set_title('MC GT', color='orange')
		plt.tight_layout()
		plt.show()