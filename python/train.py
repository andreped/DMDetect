import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow_addons as tfa
from datetime import datetime
from models import get_arch
from create_data import get_dataset


# today's date and time
today = datetime.now()
name = today.strftime("%d%m") + today.strftime("%Y")[2:] + "_" + today.strftime("%H%M%S") + "_"

# whether or not to use GPU for training (-1 == no GPU, else GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# paths
data_path = "C:/Users/andrp/workspace/DeepXDMDetect/data/DDSM_mammography_data/"
save_path = "C:/Users/andrp/workspace/DeepXDMDetect/output/models/"
history_path = "C:/Users/andrp/workspace/DeepXDMDetect/output/history/"

print(data_path + "training10_0/training10_0.tfrecords")

# PARAMS
N_SAMPLES = 55890  # https://www.kaggle.com/skooch/ddsm-mammography
N_FOLDS = 5  # 5 folds to choose from
N_EPOCHS = 100
MODEL_ARCH = 2  # which architecture/CNN to use - see models.py for info about archs
BATCH_SIZE = 16
BUFFER_SIZE = 2 ** 2
N_TRAIN_STEPS = int(N_SAMPLES / N_FOLDS / BATCH_SIZE)
N_VAL_STEPS = int(N_SAMPLES / N_FOLDS / BATCH_SIZE)
SHUFFLE_FLAG = True
instance_size = (299, 299, 1)  # Default: (299, 299, 1). Set this to (299, 299, 1) to not downsample further.
num_classes = 2
num_training_samples = 1

# get some training and validation data for building the model
train_set = get_dataset(BATCH_SIZE, data_path + "training10_0/", num_classes, SHUFFLE_FLAG, instance_size[:-1])
val_set = get_dataset(BATCH_SIZE, data_path + "training10_1/", num_classes, SHUFFLE_FLAG, instance_size[:-1])

## Model architecture (CNN)
model = get_arch(MODEL_ARCH, instance_size, num_classes)
print(model.summary())  # prints the full architecture

model.compile(
    optimizer="adam",  # most popular optimizer
    loss=tfa.losses.SigmoidFocalCrossEntropy(gamma=3),  # "categorical_crossentropy",  # because of class imbalance we use focal loss to train a model that works well on both classes
    weighted_metrics=["accuracy"],
)

save_best = ModelCheckpoint(
    filepath=save_path + name + "classifier_model.h5",
    save_best_only=True,  # only saves if model has improved (after each epoch)
    save_weights_only=False,
    verbose=1,
    monitor="val_accuracy",  # default: "val_loss" (only saves model/overwrites if val_loss has decreased)
)

history = CSVLogger(
    history_path + name + "training_history.csv",
    append=True
)

model.fit(
    train_set,
    steps_per_epoch=N_TRAIN_STEPS,
    epochs=N_EPOCHS,
    validation_data=val_set,
    validation_steps=N_VAL_STEPS,
    class_weight={0: 0.15, 1: 0.86},  # apriori, we know the distribution of the two classes, so we add a higher weight to class 1, as it is less frequent
    callbacks=[save_best, history]
)