import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
from utils import *
import h5py
import scipy
import matplotlib.pyplot as plt
from aug import augment_numpy


# https://towardsdatascience.com/overcoming-data-preprocessing-bottlenecks-with-tensorflow-data-service-nvidia-dali-and-other-d6321917f851
def get_dataset(batch_size, data_path, num_classes, shuffle=True, out_shape=(299, 299), train_mode=False):

	# parse TFRecord
	def parse_image_function(example_proto):
		image_feature_description = {
			'label': tf.io.FixedLenFeature([], tf.int64),
			'label_normal': tf.io.FixedLenFeature([], tf.int64),
			'image': tf.io.FixedLenFeature([], tf.string)
		}

		features = tf.io.parse_single_example(example_proto, image_feature_description)
		image = tf.io.decode_raw(features['image'], tf.uint8)
		image.set_shape([1 * 299 * 299])
		image = tf.reshape(image, [299, 299, 1])  # original image size is 299x299x1
		image = tf.image.grayscale_to_rgb(image)  # convert gray image to RGB image relevant for using pretrained CNNs and finetuning
		image = tf.image.resize(image, out_shape)

		if num_classes == 2:
			label = tf.cast(features['label_normal'], tf.int32)
		elif num_classes == 5:
			label = tf.cast(features['label'], tf.int32)
		elif (num_classes == [2, 5]):
			label = [tf.cast(features['label_normal'], tf.int32), tf.cast(features['label'], tf.int32)]
		elif (num_classes == [5, 2]):
			label = [tf.cast(features['label'], tf.int32), tf.cast(features['label_normal'], tf.int32)]
		else:
			print("Unvalid num_classes was given. Only valid values are {2, 5, [2, 5], [5, 2]}.")
			exit()

		if type(label) == list:
			label = {"cl" + str(i+1): tf.one_hot(label[i], num_classes[i]) for i in range(len(label))}
		else:
			label = tf.one_hot(label, num_classes)  # create one-hotted GT compatible with softmax, also convenient for multi-class... 
		
		return image, label

	# blur filter
	def blur(image, label):
		image = tfa.image.gaussian_filter2d(image=image,
							filter_shape=(11, 11), sigma=0.8)
		return image, label

	# rescale filter
	def rescale(image, label):
		image = preprocessing.Rescaling(1.0 / 255)(image)
		return image, label

	# augmentation filters
	def augment(image, label):
		'''
		data_augmentation = tf.keras.Sequential(
			[
				tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
				tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
				tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)  # Be careful doing these types of augmentations as the lesion might fall outside the image, especially for zoom and shift

			]
		)  # @TODO: Does both horizontal AND vertical make sense in this case?
		image = data_augmentation(image)
		'''
		return image, label

	autotune = tf.data.experimental.AUTOTUNE
	options = tf.data.Options()
	options.experimental_deterministic = False
	records = tf.data.Dataset.list_files(data_path, shuffle=shuffle).with_options(options)

	# load from TFRecord files
	ds = tf.data.TFRecordDataset(records, num_parallel_reads=autotune).repeat()
	ds = ds.map(parse_image_function, num_parallel_calls=autotune)
	#ds = ds.map(dilate, num_parallel_calls=autotune)
	#ds = ds.map(blur, num_parallel_calls=autotune)  # @ TODO: Should this augmentation method be mixed in with the rest of the methods? Perhaps it already exists in TF by default?
	ds = ds.batch(batch_size)
	ds = ds.map(rescale, num_parallel_calls=autotune)
	# @TODO: Something wrong here 
	if train_mode:
		#ds = ds.map(lambda image, label: (augment(image, label)), num_parallel_calls=autotune)  # only apply augmentation in training mode

		#'''
		# https://www.tensorflow.org/tutorials/images/data_augmentation#option_2_apply_the_preprocessing_layers_to_your_dataset
		# @ However, enabling augmentation seem to result in a memory leak (quite big one actually). Thus, should avoid using this for now.
		ds = ds.map(
			   lambda image, label: (tf.image.convert_image_dtype(image, tf.float32), label)
			  ).cache(  # @TODO: Is it this cache() that produces memory leak?
			  ).map(
					lambda image, label: (tf.image.random_flip_left_right(image), label)
			  ).map(
					lambda image, label: (tf.image.random_flip_up_down(image), label)
			  #).map(
			  #      lambda image, label: (tf.image.random_contrast(image, lower=0.0, upper=1.0), label)
			  )
		#'''

	ds = ds.prefetch(autotune)
	return ds



def batch_gen(file_list, batch_size, aug={}, class_names=[], input_shape=(512, 512, 1), epochs=1,
			  mask_flag=False, fine_tune=False, inference_mode=False):
	while True:  # <- necessary for end of training (last epoch)
		for i in range(epochs):
			batch = 0
			nb_classes = len(class_names) + 1
			class_names = np.array(class_names)

			# shuffle samples for each epoch
			np.random.shuffle(file_list)  # patients are shuffled, but chunks are after each other

			input_batch = []
			output_batch = []

			for filename in file_list:

				# read whole volume as an array
				with h5py.File(filename, 'r') as f:
					data = np.expand_dims(np.array(f["data"]).astype(np.float32), axis=-1)
					output = []
					for class_ in class_names:
						output.append(np.expand_dims(np.array(f[class_]).astype(np.float32), axis=-1))
					output = np.concatenate(output, axis=-1)

				# need to filter all classes of interest within "_mammary_gland" away from the gland class
				if ("_pectoral_muscle" in class_names) and (nb_classes > 2):
					tmp1 = output[..., np.argmax(class_names == "_pectoral_muscle")]
					for c in class_names:
						if c != "_pectoral_muscle":
							tmp2 = output[..., np.argmax(class_names == c)]
							tmp2[tmp1 == 1] = 0
							output[..., np.argmax(class_names == c)] = tmp2

				# filter "_cancer" class away from all other relevant classes
				if ("_cancer" in class_names) and (nb_classes > 2):
					tmp1 = output[..., np.argmax(class_names == "_cancer")]
					for c in class_names:
						if c != "_cancer":
							tmp2 = output[..., np.argmax(class_names == c)]
							tmp2 = np.clip(tmp2 - tmp1, a_min=0, a_max=1)
							output[..., np.argmax(class_names == c)] = tmp2

				# filter "_cancer" class away from all other relevant classes
				if ("_nipple" in class_names) and (nb_classes > 2):
					tmp1 = output[..., np.argmax(class_names == "_nipple")]
					for c in class_names:
						if c != "_nipple":
							tmp2 = output[..., np.argmax(class_names == c)]
							tmp2 = np.clip(tmp2 - tmp1, a_min=0, a_max=1)
							output[..., np.argmax(class_names == c)] = tmp2

				# filter "_cancer" class away from all other relevant classes
				if ("_thick_vessels" in class_names) and (nb_classes > 2):
					tmp1 = output[..., np.argmax(class_names == "_thick_vessels")]
					for c in class_names:
						if c != "_thick_vessels":
							tmp2 = output[..., np.argmax(class_names == c)]
							tmp2 = np.clip(tmp2 - tmp1, a_min=0, a_max=1)
							output[..., np.argmax(class_names == c)] = tmp2

				# add background class to output
				tmp = np.sum(output, axis=-1)
				tmp = (tmp == 0).astype(np.float32)
				output = np.concatenate([np.expand_dims(tmp, axis=-1), output], axis=-1)

				# augment
				data, output = augment_numpy(data, output, aug)

				# intensity normalize (0, 255) => (0, 1)
				data /= 255.

				input_batch.append(data)
				output_batch.append(output)

				batch += 1
				if batch == batch_size:
					# reset and yield
					batch = 0
					x_ = np.array(input_batch)
					y_ = np.array(output_batch)
					input_batch = []
					output_batch = []

					if inference_mode:
						yield filename, (x_, y_)
					else:
						yield x_, y_