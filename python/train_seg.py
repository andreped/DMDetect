import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow_addons as tfa
from datetime import datetime
from models import get_arch, Unet
from batch_generator import get_dataset, batch_gen
from utils import macro_accuracy, flatten_
from tensorflow.keras.optimizers import Adam
from resunetpp import *
# from accumulated_gradients import AccumOptimizer  # @TODO: Currently, these accumulated gradients solutions are not compatible with something in TF 2


# today's date and time
today = datetime.now()
name = today.strftime("%d%m") + today.strftime("%Y")[2:] + "_" + today.strftime("%H%M%S") + "_"

# whether or not to use GPU for training (-1 == no GPU, else GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
'''

# paths
data_path =  "../data/CSAW-S_preprocessed_1024_True/"  # "..data/CSAW-S/CSAW-S/CsawS/anonymized_dataset/" #"../data/DDSM_mammography_data/"
save_path = "../output/models/"
history_path = "../output/history/"

# PARAMS
N_EPOCHS = 1000  # 200
batch_size = 2  # 12, 16
accum_steps = 4  # number of steps when performing accumulated gradients
SHUFFLE_FLAG = True
img_size = int(data_path.split("_")[-2])  # 512
fine_tune = 1  # if set to 1, does not perform fine-tuning
input_shape = (img_size, img_size, fine_tune)  # Default: (299, 299, 1). Set this to (299, 299, 1) to not downsample further.
learning_rate = 1e-3  # relevant for the optimizer, Adam used by default (with default lr=1e-3), I normally use 1e-4 when finetuning
gamma = 3  # Focal Loss parameter
AUG_FLAG = False  # Whether or not to apply data augmentation during training (only applied to the training set)
train_aug = {"horz": 1, "gamma": [0.75, 1.5]}  # {"vert": 1, "horz": 1, "rot90": 1, "gamma": [0.75, 1.5]}
val_aug = {}
spatial_dropout = 0.1  # 0.1
N_PATIENTS = 150
train_val_split = 0.8
use_background = True  # False  (will neglect background class if False)
model_arch = "unet"  # {"unet", "resunetpp"}
#renorm = True  # False (whether to apply BatchReNormalization in U-Net)

'''
class_names = [
    "_axillary_lymph_nodes", "_calcifications", "_cancer",\
    "_foreign_object", "_mammary_gland", "_nipple",\
    "_non-mammary_tissue", "_pectoral_muscle", "_skin", "_text",\
    "_thick_vessels", "_unclassified"
]
'''

# @FIXME: pectoral_muscle/mammary_gland has not been consistently annotated
class_names = [
    "_cancer", "_mammary_gland", "_pectoral_muscle", "_nipple",  # "_skin", "_thick_vessels"
]
nb_classes = len(class_names) + 1  # include background class (+1)

# add hyperparams to name of session, to be easier to parse during eval and overall
name += "bs_" + str(batch_size) + "_arch_" + model_arch + "_img_" + str(img_size) + "_nbcl_" +\
 str(nb_classes) + "_gamma_" + str(gamma) + "_aug_" +\
 str(list(train_aug)).replace("'", "").replace("[", "").replace("]", "").replace(" ", "") +\
 "_drp_" + str(spatial_dropout).replace(".", ",") 
#+ "_renorm_" + str(renorm)
name += "_"

print("\nCurrent model run: ")
print(name, "\n")

## create train and validation sets
data_set = []
for patient in os.listdir(data_path):
    curr = data_path + patient + "/"
    tmp = [curr + x for x in os.listdir(curr)]
    data_set.append(tmp)

val1 = int(N_PATIENTS * train_val_split)
train_set = data_set[:val1]
val_set = data_set[val1:]

train_set = flatten_(train_set)
val_set = flatten_(val_set)

N_TRAIN_STEPS = int(np.ceil(len(train_set) / batch_size))
N_VAL_STEPS = int(np.ceil(len(val_set) / batch_size))


# define data generators
train_gen = batch_gen(train_set, batch_size, aug=train_aug, class_names=class_names, input_shape=input_shape, epochs=N_EPOCHS, mask_flag=False, fine_tune=False)
val_gen = batch_gen(val_set, batch_size, aug=val_aug, class_names=class_names, input_shape=input_shape, epochs=N_EPOCHS, mask_flag=False, fine_tune=False)

# define model
if model_arch == "unet":
    network = Unet(input_shape=input_shape, nb_classes=nb_classes)
    network.encoder_spatial_dropout = spatial_dropout  # attempt to remove spatial dropout to see if it improves the issue with faulty classes...
    network.decoder_spatial_dropout = spatial_dropout  #  - Spatial Dropout extremely important to get good generalization and keep model learning what it should!
    #network.set_convolutions([8, 16, 32, 32, 64, 64, 128, 256, 128, 64, 64, 32, 32, 16, 8])
    network.set_convolutions([16, 32, 32, 64, 64, 128, 128, 256, 128, 128, 64, 64, 32, 32, 16])
    if img_size == 1024:
        network.set_bottom_level(8)
    # network.set_renorm(renorm)
    model = network.create()
elif model_arch == "resunetpp":
    network = ResUnetPlusPlus(input_shape=input_shape, nb_classes=nb_classes)
    # network.set_convolutions([16, 32, 64, 128, 256, 512])  # [16, 32, 64, 128, 256]  # suitable for 256x256 input
    network.set_convolutions([16, 32, 64, 128, 256, 512])  # attempt to make it more shallow => perhaps won't overfit so easily? Perhaps I sould just use dropout
    model = network.create()
else:
    print("Unknown architecture selected. Please choose one of these: {'unet', 'resunet++'}")
    exit()
print(model.summary())  # prints the full architecture

#opt = AccumOptimizer(Adam(lr=learning_rate), steps_per_update=accum_steps)

model.compile(
    #optimizer=opt,
    optimizer=Adam(learning_rate),
    loss=network.get_dice_loss(use_background=use_background)
)

save_best = ModelCheckpoint(
    filepath=save_path + name + "segmentation_model.h5",
    save_best_only=True,  # only saves if model has improved (after each epoch)
    save_weights_only=False,
    verbose=1,
    monitor="val_loss",  # "val_f1_score",  # default: "val_loss" (only saves model/overwrites if val_loss has decreased)
    mode="min",  # default 'auto', but using custom losses it might be necessary to set it to 'max', as it is interpreted to be minimized by default, is unknown
)

history = CSVLogger(
    history_path + name + "training_history.csv",
    append=True
)

model.fit(
    train_gen,
    steps_per_epoch=N_TRAIN_STEPS,
    epochs=N_EPOCHS,
    validation_data=val_gen,
    validation_steps=N_VAL_STEPS,
    callbacks=[save_best, history]
)