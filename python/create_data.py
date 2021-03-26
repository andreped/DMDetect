import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing


# https://towardsdatascience.com/overcoming-data-preprocessing-bottlenecks-with-tensorflow-data-service-nvidia-dali-and-other-d6321917f851
def get_dataset(batch_size, data_path, num_classes, shuffle=True, out_shape=(299, 299)):
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
        image = tf.reshape(image, [299, 299, 1])
        image = tf.image.resize(image, out_shape)
        label = tf.cast(features['label_normal'], tf.int32)
        label = tf.one_hot(label, num_classes)  # create one-hotted GT compatible with softmax, also convenient for multi-class...
        return image, label

    # rescale filter
    def rescale(image, label):
        image = preprocessing.Rescaling(1.0 / 255)(image)
        return image, label

    autotune = tf.data.experimental.AUTOTUNE
    options = tf.data.Options()
    options.experimental_deterministic = False
    records = tf.data.Dataset.list_files(data_path + '*', shuffle=shuffle).with_options(options)

    # load from TFRecord files
    ds = tf.data.TFRecordDataset(records, num_parallel_reads=autotune).repeat()
    ds = ds.map(parse_image_function, num_parallel_calls=autotune)
    #ds = ds.map(dilate, num_parallel_calls=autotune)
    #ds = ds.map(blur, num_parallel_calls=autotune)
    ds = ds.batch(batch_size)
    ds = ds.map(rescale,num_parallel_calls=autotune)
    #ds = ds.map(augment, num_parallel_calls=autotune)
    ds = ds.prefetch(autotune)
    return ds