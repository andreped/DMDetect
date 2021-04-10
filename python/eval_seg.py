import numpy as np
import os
from batch_generator import get_dataset
from tensorflow.keras.models import load_model
from tqdm import tqdm
from sklearn.metrics import classification_report, auc, roc_curve
import matplotlib.pyplot as plt
from tf_explain.core import GradCAM, IntegratedGradients
import tensorflow as tf
from utils import flatten_, DSC, IOU, argmax_keepdims, one_hot_fix, post_process, random_jet_colormap, make_subplots, post_process_mammary_gland
from batch_generator import batch_gen
import cv2
from prettytable import PrettyTable, MARKDOWN


# change print precision
#np.set_printoptions(precision=4)
np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})


# whether or not to use GPU for training (-1 == no GPU, else GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # set to 0 to use GPU

# allow growth, only use the GPU memory required to solve a specific task (makes room for doing stuff in parallel)
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

# paths
data_path =  "../data/"  # CSAW-S_preprocessed_1024_True/"  # "..data/CSAW-S/CSAW-S/CsawS/anonymized_dataset/" #"../data/DDSM_mammography_data/"
save_path = "../output/models/"
history_path = "../output/history/"

name = "030421_003122_bs_12_arch_unet_imgsize_512_nbcl_7_gamma_3_segmentation_model"  # best
name = "030421_152539_bs_12_arch_unet_imgsize_512_nbcl_6_gamma_3_segmentation_model"  # tested new spatial dropout scheme in U-Net
name = "030421_185828_bs_12_arch_unet_imgsize_512_nbcl_6_gamma_3_segmentation_model"  # with aug
name = "030421_204721_bs_12_arch_unet_img_512_nbcl_5_gamma_3_aug_vert,horz,rot90,gamma_segmentation_model"  # reduced gamma aug and removed skin class
#name = "030421_214156_bs_12_arch_unet_img_512_nbcl_5_gamma_3_aug_horz-gamma_segmentation_model"
#name = "030421_235137_bs_4_arch_resunetpp_img_512_nbcl_5_gamma_3_aug_horz,gamma_drp_0.2_segmentation_model"  # new unmodified ResUNet++ model 
#name = "040421_170033_bs_4_arch_resunetpp_img_512_nbcl_5_gamma_3_aug_horz,gamma_drp_0,2_segmentation_model"  # 
#name = "040421_195756_bs_8_arch_unet_img_512_nbcl_5_gamma_3_aug_horz,gamma_drp_0,1_segmentation_model"  # U-Net + spatial dropout 0.1 + batch size 8 (BEST SO FAR!)
name = "050421_021022_bs_2_arch_unet_img_1024_nbcl_5_gamma_3_aug_horz,gamma_drp_0,1_segmentation_model"  # (BEST) 1024 input, batch size 8, U-Net (struggled to converge, didnt overfit), however, MUCH better performance on tumour class and slightly better overall, (only slightly worse on nipple class, but likely not significantly)

print("\nCurrent model name:")
print(name)

img_size = int(name.split("img_")[-1].split("_")[0])
data_path += "CSAW-S_preprocessed_" + str(img_size) + "_True/"

plot_flag = False  # False
N_PATIENTS = 150
train_val_split = 0.8
#img_size = int(data_path.split("_")[-2])
input_shape = (512, 512, 1)
class_names = ["_cancer", "_mammary_gland", "_pectoral_muscle", "_nipple"]  # ["_cancer", "_mammary_gland", "_pectoral_muscle"]  # has to be updated dependend on which model "name" is used
nb_classes = len(class_names) + 1

all_class_names = np.array(["background"] + [x[1:] for x in class_names])

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

chosen_set = val_set

# define data generator
generator = batch_gen(chosen_set, batch_size=1, aug={}, class_names=class_names, input_shape=input_shape, epochs=1, mask_flag=False, fine_tune=False, inference_mode=True)

# load trained model (for deployment or usage in diagnostics - freezed, thus deterministic, at least in theory)
model = load_model(save_path + name + ".h5", compile=False)

# random colormap
some_cmap = random_jet_colormap()

dsc_ = []
iou_ = []

dsc_classes_ = []
iou_classes_ = []

cnt = 0
for filename, (x, y) in tqdm(generator, "DM: ", total=len(chosen_set)):
	conf = model.predict(x)
	pred = np.argmax(conf, axis=-1)
	# print(np.unique(pred))
	pred = one_hot_fix(pred, nb_classes)

	# post-process pred and GT to match original image size for proper evaluation
	# print(filename)
	tmp = filename.split("/")[-1].split(".")[0].split("_")
	orig_shape = (int(tmp[1]), int(tmp[2]))
	new_shape = (int(tmp[3]), int(tmp[4]))

	'''
	if plot_flag:
		x_orig = post_process(np.squeeze(x, axis=0), new_shape, orig_shape, resize=True, interpolation=cv2.INTER_LINEAR)
		y_orig = post_process(np.squeeze(y, axis=0), new_shape, orig_shape, resize=True, interpolation=cv2.INTER_LINEAR)
		pred_orig = post_process(np.squeeze(pred, axis=0), new_shape, orig_shape, resize=True, interpolation=cv2.INTER_LINEAR)
	'''

	## first post processing to go back to original shape
	x = post_process(np.squeeze(x, axis=0), new_shape, orig_shape, resize=False, interpolation=cv2.INTER_LINEAR)
	y = post_process(np.squeeze(y, axis=0), new_shape, orig_shape, resize=False, interpolation=cv2.INTER_LINEAR)
	pred = post_process(np.squeeze(pred, axis=0), new_shape, orig_shape, resize=False, interpolation=cv2.INTER_LINEAR)
	conf = post_process(np.squeeze(conf, axis=0), new_shape, orig_shape, resize=False, interpolation=cv2.INTER_LINEAR)

	## second post processing to fix mammary gland prediction
	pred = post_process_mammary_gland(pred, all_class_names)

	if plot_flag:
		make_subplots(x, y, pred, conf, img_size, all_class_names, some_cmap)

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
	if cnt == len(chosen_set) + 1:
		break


dsc_ = np.array(dsc_)
iou_ = np.array(iou_)
dsc_classes_ = np.array(dsc_classes_)
iou_classes_ = np.array(iou_classes_)

x = PrettyTable()
x.field_names = ["metrics"] + [x[1:].replace("_", " ") for x in class_names] + ["overall"]
x.add_row(["DSC"] + list(np.mean(dsc_classes_, axis=0)) + [np.mean(dsc_, axis=0)])
x.add_row(["IOU"] + list(np.mean(iou_classes_, axis=0)) + [np.mean(iou_, axis=0)])
x.set_style(MARKDOWN)
x.float_format = ".3"

print("Print table summary of the results: ")
print(x)