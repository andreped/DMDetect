import numpy as np
import os
from batch_generator import get_dataset
from tensorflow.keras.models import load_model
from tqdm import tqdm
from sklearn.metrics import classification_report, auc, roc_curve
import matplotlib.pyplot as plt
from tf_explain.core import GradCAM, IntegratedGradients
import tensorflow as tf
from utils import flatten_, DSC, IOU, argmax_keepdims, one_hot_fix
from batch_generator import batch_gen


# change print precision
#np.set_printoptions(precision=4)
np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})


# whether or not to use GPU for training (-1 == no GPU, else GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # set to 0 to use GPU

# allow growth, only use the GPU memory required to solve a specific task (makes room for doing stuff in parallel)
physical_devices = tf.config.list_physical_devices('GPU')
try:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
	# Invalid device or cannot modify virtual devices once initialized.
	pass

# paths
data_path =  "../data/CSAW-S_preprocessed/"  # "..data/CSAW-S/CSAW-S/CsawS/anonymized_dataset/" #"../data/DDSM_mammography_data/"
save_path = "../output/models/"
history_path = "../output/history/"

name = "020421_212113_bs_8_arch_unet_imgsize_512_nbcl_7_gamma_3_segmentation_model"

N_PATIENTS = 150
train_val_split = 0.8
input_shape = (512, 512, 1)
class_names = ["_cancer", "_mammary_gland", "_pectoral_muscle", "_skin", "_thick_vessels", "_nipple"]  # ["_cancer", "_mammary_gland", "_pectoral_muscle"]  # has to be updated dependend on which model "name" is used
nb_classes = len(class_names) + 1

# create test set
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

# define data generator
generator = batch_gen(val_set, batch_size=1, aug={}, class_names=class_names, input_shape=input_shape, epochs=1, mask_flag=False, fine_tune=False)

# load trained model (for deployment or usage in diagnostics - freezed, thus deterministic, at least in theory)
model = load_model(save_path + name + ".h5", compile=False)

dsc_ = []
iou_ = []

dsc_classes_ = []
iou_classes_ = []

cnt = 0
for x, y in tqdm(generator, "DM: ", total=len(val_set)):
	pred = model.predict(x)
	pred = np.argmax(pred, axis=-1)
	print(np.unique(pred))
	pred = one_hot_fix(pred, nb_classes)

	'''
	fig, ax = plt.subplots(nb_classes, 3)
	ax[0, 0].imshow(x[0], cmap="gray")

	for i in range(nb_classes):
		ax[i, 1].imshow(pred[0, ..., i], cmap="gray")
		ax[i, 2].imshow(y[0, ..., i], cmap="gray")
	plt.show()
	'''

	# per-class (micro) DSC and IOU
	tmp1 = []
	tmp2 = []
	for c in range(1, nb_classes):
		tmp1.append(DSC(np.expand_dims(y[..., c], axis=-1), np.expand_dims(pred[..., c], axis=-1), remove_bg=False))
		tmp2.append(IOU(np.expand_dims(y[..., c], axis=-1), np.expand_dims(pred[..., c], axis=-1), remove_bg=False))
	dsc_classes_.append(tmp1)
	iou_classes_.append(tmp2)

	# overall DSC and IOU (macro-averaged)
	dsc_curr = DSC(y, pred)
	iou_curr = IOU(y, pred)

	dsc_.append(dsc_curr)
	iou_.append(iou_curr)

	cnt += 1
	if cnt == len(val_set)+1:
		break


dsc_ = np.array(dsc_)
iou_ = np.array(iou_)
dsc_classes_ = np.array(dsc_classes_)
iou_classes_ = np.array(iou_classes_)

print("For the classes: ")
print(class_names)
print("DSC: ", np.mean(dsc_classes_, axis=0))
print("IOU: ", np.mean(iou_classes_, axis=0))
print("Overall (DSC and IOU): ", np.mean(dsc_, axis=0), np.mean(iou_, axis=0))

# get summary statistics (performance metrics)
#summary = classification_report(ps, gs)
#print(summary)