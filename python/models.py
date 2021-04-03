import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten, SpatialDropout2D, \
    ZeroPadding2D, Activation, AveragePooling2D, UpSampling2D, BatchNormalization, ConvLSTM2D, \
    TimeDistributed, Concatenate, Lambda, Reshape
from tensorflow.keras.models import Model, Sequential


# see here for already built-in pretrained architectures:
# https://keras.io/api/applications/

def get_arch(MODEL_ARCH, instance_size, num_classes):

    # basic
    if MODEL_ARCH == 1:
        # define model (some naive network)
        model = Sequential()  # example of creation of TF-Keras model using the Sequential
        model.add(Conv2D(32, kernel_size=(3, 3), input_shape=instance_size))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Activation('relu'))
        model.add(Dense(num_classes, activation='softmax'))

    elif MODEL_ARCH == 2:
        # InceptionV3 (typical example arch) - personal preference for CNN classification (however, quite expensive and might be overkill in a lot of scenarios)
        some_input = Input(shape=instance_size)
        base_model = InceptionV3(include_top=False, weights="imagenet", pooling=None, input_tensor=some_input)
        x = base_model.output
        x = Flatten()(x)
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Activation('relu')(x)
        x = Dense(num_classes, activation='softmax')(x) 
        model = Model(inputs=base_model.input, outputs=x)  # example of creation of TF-Keras model using the functional API

    elif MODEL_ARCH == 3:
        # ResNet-50, another very popular arch, can be done similarly as for InceptionV3 above
        some_input = Input(shape=instance_size)
        base_model = ResNet50(include_top=False, weights="imagenet", pooling=None, input_tensor=some_input)
        x = base_model.output
        x = Flatten()(x)
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Activation('relu')(x)
        x = Dense(num_classes, activation='softmax')(x) 
        model = Model(inputs=base_model.input, outputs=x)

    elif MODEL_ARCH == 4:
        # Example of a multi-task model, performing both binary classification AND multi-class classification simultaneously, distinguishing
        # normal tissue from breast cancer tissue, as well as separating different types of breast cancer tissue
        some_input = Input(shape=instance_size)
        base_model = InceptionV3(include_top=False, weights="imagenet", pooling=None, input_tensor=some_input)
        x = base_model.output
        x = Flatten()(x)

        # first output branch
        y1 = Dense(64)(x)
        y1 = BatchNormalization()(y1)
        y1 = Dropout(0.5)(y1)
        y1 = Activation('relu')(y1)
        y1 = Dense(num_classes[0], activation='softmax', name="cl1")(y1)

        # second output branch
        y2 = Dense(64)(x)
        y2 = BatchNormalization()(y2)
        y2 = Dropout(0.5)(y2)
        y2 = Activation('relu')(y2)
        y2 = Dense(num_classes[1], activation='softmax', name="cl2")(y2)

        model = Model(inputs=base_model.input, outputs=[y1, y2])  # example of multi-task network through the functional API

    else:
        print("please choose supported models: {1, 2, 3, 4}")
        exit()

    return model


def convolution_block_2d(x, nr_of_convolutions, use_bn=False, spatial_dropout=None):
    for i in range(2):
        x = Convolution2D(nr_of_convolutions, 3, padding='same')(x)
        if use_bn:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if spatial_dropout:
            x = SpatialDropout2D(spatial_dropout)(x)

    return x


def encoder_block_2d(x, nr_of_convolutions, use_bn=False, spatial_dropout=None):

    x_before_downsampling = convolution_block_2d(x, nr_of_convolutions, use_bn, spatial_dropout)
    x = MaxPooling2D((2, 2))(x_before_downsampling)

    return x, x_before_downsampling


def decoder_block_2d(x, nr_of_convolutions, cross_over_connection=None, use_bn=False, spatial_dropout=None):

    x = UpSampling2D((2, 2))(x)
    if cross_over_connection is not None:
        x = Concatenate()([cross_over_connection, x])
    x = convolution_block_2d(x, nr_of_convolutions, use_bn, spatial_dropout)

    return x


def encoder_block(x, nr_of_convolutions, use_bn=False, spatial_dropout=None, dims=2):
    if dims == 2:
        return encoder_block_2d(x, nr_of_convolutions, use_bn, spatial_dropout)
    else:
        raise ValueError


def decoder_block(x, nr_of_convolutions, cross_over_connection=None, use_bn=False, spatial_dropout=None, dims=2):
    if dims == 2:
        return decoder_block_2d(x, nr_of_convolutions, cross_over_connection, use_bn, spatial_dropout)
    else:
        raise ValueError


def convolution_block(x, nr_of_convolutions, use_bn=False, spatial_dropout=None, dims=2):
    if dims == 2:
        return convolution_block_2d(x, nr_of_convolutions, use_bn, spatial_dropout)
    else:
        raise ValueError


class Unet():
    def __init__(self, input_shape, nb_classes):
        if len(input_shape) != 3 and len(input_shape) != 4:
            raise ValueError('Input shape must have 3 or 4 dimensions')
        if len(input_shape) == 3:
            self.dims = 2
        else:
            self.dims = 3
        if nb_classes <= 1:
            raise ValueError('Segmentation classes must be > 1')
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.convolutions = None
        self.encoder_use_bn = True
        self.decoder_use_bn = True
        self.encoder_spatial_dropout = None
        self.decoder_spatial_dropout = None
        self.bottom_level = 4
        self.dropout_level_threshold = 1

    def set_convolutions(self, convolutions):
        if len(convolutions) != self.get_depth()*2 + 1:
            raise ValueError('Nr of convolutions must have length ' + str(self.get_depth()*2 + 1))
        self.convolutions = convolutions

    def get_depth(self):
        init_size = max(self.input_shape[:-1])
        size = init_size
        depth = 0
        while size % 2 == 0 and size > self.bottom_level:
            size /= 2
            depth += 1

        return depth

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


    def create(self):
        """
        Create model and return it
        :return: keras model
        """

        input_layer = Input(shape=self.input_shape)
        x = input_layer

        init_size = max(self.input_shape[:-1])
        size = init_size

        convolutions = self.convolutions
        if convolutions is None:
            # Create convolutions
            convolutions = []
            nr_of_convolutions = 8
            for i in range(self.get_depth()):
                convolutions.append(int(nr_of_convolutions))
                nr_of_convolutions *= 2
            convolutions.append(int(nr_of_convolutions))
            for i in range(self.get_depth()):
                convolutions.append(int(nr_of_convolutions))
                nr_of_convolutions /= 2

        depth = self.get_depth()

        curr_encoder_spatial_dropout = self.encoder_spatial_dropout
        curr_decoder_spatial_dropout = self.decoder_spatial_dropout

        connection = {}
        i = 0
        while size % 2 == 0 and size > self.bottom_level:
            if i < self.dropout_level_threshold:  # only apply dropout at the bottom (deep features)
                curr_encoder_spatial_dropout = None
            else:
                curr_encoder_spatial_dropout = self.encoder_spatial_dropout
            x, connection[size] = encoder_block(x, convolutions[i], self.encoder_use_bn, curr_encoder_spatial_dropout, self.dims)
            size /= 2
            i += 1

        x = convolution_block(x, convolutions[i], self.encoder_use_bn, curr_encoder_spatial_dropout, self.dims)
        i += 1

        steps = int(i)
        j = 0
        while size < init_size:
            if steps - j - 1 <= self.dropout_level_threshold:  # only apply dropout at the bottom (deep features)
                curr_decoder_spatial_dropout = None
            else:
                curr_decoder_spatial_dropout = self.decoder_spatial_dropout
            size *= 2
            x = decoder_block(x, convolutions[i], connection[size], self.decoder_use_bn, curr_decoder_spatial_dropout, self.dims)
            i += 1
            j += 1

        if self.dims == 2:
            x = Convolution2D(self.nb_classes, 1, activation='softmax')(x)
        else:
            raise ValueError

        return Model(inputs=input_layer, outputs=x)