import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing


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
        data_augmentation = tf.keras.Sequential(
           [preprocessing.RandomFlip("horizontal_and_vertical"),  # @TODO: Does both horizontal AND vertical make sense in this case?
            preprocessing.RandomRotation(0.1),
            preprocessing.RandomZoom(0.1)])  # Be careful doing these types of augmentations as the lesion might fall outside the image, especially for zoom and shift
        image = data_augmentation(image)
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
    #if train_mode:
    #   ds = ds.map(augment, num_parallel_calls=autotune)  # only apply augmentation in training mode
    ds = ds.prefetch(autotune)
    return ds