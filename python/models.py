import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.models import load_model, Model, Sequential


# see here for already built-in pretrained architectures:
# https://keras.io/api/applications/

def get_arch(MODEL_ARCH, instance_size, num_classes):

    # basic
    if MODEL_ARCH == 1:
        # define model
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
        # InceptionV3 (typical example arch)
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
        x = Dense(128)(x)
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
        print("please choose supported models: {0, 1}")
        exit()

    return model