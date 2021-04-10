"""
ResUNet++ architecture in Keras TensorFlow
Based on implementation from:
https://github.com/DebeshJha/ResUNetPlusPlus/blob/master/m_resunet.py
"""
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    #print()
    #print(init, init.shape, init[channel_axis])
    filters = init.shape[channel_axis]  # .value
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x

def stem_block(x, n_filter, strides):
    x_init = x

    ## Conv 1
    x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same")(x)

    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
    s = BatchNormalization()(s)

    ## Add
    x = Add()([x, s])
    x = squeeze_excite_block(x)
    return x


def resnet_block(x, n_filter, strides=1):
    x_init = x

    ## Conv 1
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
    ## Conv 2
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(n_filter, (3, 3), padding="same", strides=1)(x)

    ## Shortcut
    s  = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
    s = BatchNormalization()(s)

    ## Add
    x = Add()([x, s])
    x = squeeze_excite_block(x)
    return x

def aspp_block(x, num_filters, rate_scale=1):
    x1 = Conv2D(num_filters, (3, 3), dilation_rate=(6 * rate_scale, 6 * rate_scale), padding="SAME")(x)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(num_filters, (3, 3), dilation_rate=(12 * rate_scale, 12 * rate_scale), padding="SAME")(x)
    x2 = BatchNormalization()(x2)

    x3 = Conv2D(num_filters, (3, 3), dilation_rate=(18 * rate_scale, 18 * rate_scale), padding="SAME")(x)
    x3 = BatchNormalization()(x3)

    x4 = Conv2D(num_filters, (3, 3), padding="SAME")(x)
    x4 = BatchNormalization()(x4)

    y = Add()([x1, x2, x3, x4])
    y = Conv2D(num_filters, (1, 1), padding="SAME")(y)
    return y

def attention_block(g, x):
    """
        g: Output of Parallel Encoder block
        x: Output of Previous Decoder block
    """

    filters = x.shape[-1]  # .value

    g_conv = BatchNormalization()(g)
    g_conv = Activation("relu")(g_conv)
    g_conv = Conv2D(filters, (3, 3), padding="SAME")(g_conv)

    g_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(g_conv)

    x_conv = BatchNormalization()(x)
    x_conv = Activation("relu")(x_conv)
    x_conv = Conv2D(filters, (3, 3), padding="SAME")(x_conv)

    gc_sum = Add()([g_pool, x_conv])

    gc_conv = BatchNormalization()(gc_sum)
    gc_conv = Activation("relu")(gc_conv)
    gc_conv = Conv2D(filters, (3, 3), padding="SAME")(gc_conv)

    gc_mul = Multiply()([gc_conv, x])
    return gc_mul

def decoder_block(c3, b1, n_filter):
    d1 = attention_block(c3, b1)
    d1 = UpSampling2D((2, 2))(d1)
    d1 = Concatenate()([d1, c3])
    return resnet_block(d1, n_filter)


class ResUnetPlusPlus:
    def __init__(self, input_shape=(256, 256, 3), nb_classes=2):
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.dims = 2  # 2D (image input)
        self.convolutions = [16, 32, 64, 128, 256]  # suitable for (256x256) input (DEFAULT FOR THE ORIGINAL IMPLEMENTATION)
        #self.convolutions = [16, 32, 64, 128, 256, 512]  # suitable for (512x512) input

    def get_dice_loss(self, use_background=False):
        def dice_loss(target, output, epsilon=1e-10):
            smooth = 1.
            dice = 0

            for object in range(0 if use_background else 1, self.nb_classes):
                if self.dims == 2:
                    output1 = output[:, :, :, object]
                    target1 = target[:, :, :, object]
                else:
                    output1 = output[:, :, :, :, object]
                    target1 = target[:, :, :, :, object]
                intersection1 = tf.reduce_sum(output1 * target1)
                union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(target1 * target1)
                dice += (2. * intersection1 + smooth) / (union1 + smooth)

            if use_background:
                dice /= self.nb_classes
            else:
                dice /= (self.nb_classes - 1)

            return tf.clip_by_value(1. - dice, 0., 1. - epsilon)

        return dice_loss

    def set_convolutions(self, convolutions):
        self.convolutions = convolutions

    def create(self):
        n_filters = self.convolutions  # [16, 32, 64, 128, 256]  # suitable for 256x256 input
        inputs = Input(self.input_shape)

        c0 = inputs
        c1 = stem_block(c0, n_filters[0], strides=1)

        ## Encoder
        cc = [c1]
        for i in range(1, len(self.convolutions)-1):
        	cc.append(resnet_block(cc[-1], n_filters[i], strides=2))

        '''
        c2 = resnet_block(c1, n_filters[1], strides=2)
        c3 = resnet_block(c2, n_filters[2], strides=2)
        c4 = resnet_block(c3, n_filters[3], strides=2)
        c5 = resnet_block(c4, n_filters[4], strides=2)
        '''

        ## Bridge
        b1 = aspp_block(cc[-1], n_filters[-1])
        #b1 = aspp_block(c5, n_filters[5])

        ## Decoder
        dd = [b1]
        for i in range(len(self.convolutions)-2, 0, -1):
        	dd.append(decoder_block(cc[i-1], dd[-1], n_filters[i]))
        
        '''
        d1 = decoder_block(c4, b1, n_filters[4])
        d2 = decoder_block(c3, d1, n_filters[3])
        d3 = decoder_block(c2, d2, n_filters[2])
        d4 = decoder_block(c1, d3, n_filters[1])
        '''

        '''
        d1 = attention_block(c3, b1)
        d1 = UpSampling2D((2, 2))(d1)
        d1 = Concatenate()([d1, c3])
        d1 = resnet_block(d1, n_filters[3])

        d2 = attention_block(c2, d1)
        d2 = UpSampling2D((2, 2))(d2)
        d2 = Concatenate()([d2, c2])
        d2 = resnet_block(d2, n_filters[2])

        d3 = attention_block(c1, d2)
        d3 = UpSampling2D((2, 2))(d3)
        d3 = Concatenate()([d3, c1])
        d3 = resnet_block(d3, n_filters[1])
        '''

        ## output
        outputs = aspp_block(dd[-1], n_filters[0])
        outputs = Conv2D(self.nb_classes, (1, 1), padding="same")(outputs)
        outputs = Activation("softmax")(outputs)

        ## Model
        model = Model(inputs, outputs)
        return model