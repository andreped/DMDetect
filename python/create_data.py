import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import tensorflow_addons as tfa
from utils import *
import h5py
import cv2
import imutils
from tqdm import tqdm
import matplotlib.pyplot as plt


def preprocess_segmentation_samples():

	data_path = "../data/CSAW-S/CSAW-S/CsawS/anonymized_dataset/"
	save_path = "../data/CSAW-S_preprocessed"

	classes_ = [
		"", "_axillary_lymph_nodes", "_calcifications", "_cancer",\
		"_foreign_object", "_mammary_gland", "_nipple",\
		"_non-mammary_tissue", "_pectoral_muscle", "_skin", "_text",\
		"_thick_vessels", "_unclassified"
	]

	id_ = "_mammary_gland"
	# scale_ = 2560 / 3328  # width / height
	img_size = 1024  # output image size (512 x 512), keep aspect ratio
	clahe_flag = True  # True

	save_path += "_" + str(img_size) + "_" + str(clahe_flag) + "/"

	if not os.path.exists(save_path):
		os.makedirs(save_path)

	for patient in tqdm(os.listdir(data_path), "DM: "):
		curr_path = data_path + patient + "/"

		patient_save_path = save_path + patient + "/"
		if not os.path.exists(patient_save_path):
			os.makedirs(patient_save_path)

		# get scans in patient folder
		scans = []
		for file_ in os.listdir(curr_path):
			if id_ in file_:
				scans.append(file_.split(id_)[0])

		# for each scan in patient, extract relevant data in .h5 file
		for scan in scans:
			scan_id = scan.split("_")[1]

			create_save_flag = True

			for class_ in classes_:
				# read image and resize (but keep aspect ratio)
				img = cv2.imread(curr_path + scan + class_ + ".png", 0)  # uint8
				orig_shape = img.shape
				img = imutils.resize(img, height=img_size)  # uint8
				new_shape = img.shape

				if create_save_flag:
					f = h5py.File(patient_save_path + scan_id + "_" + str(orig_shape[0]) + "_" + str(orig_shape[1]) +\
						"_" + str(new_shape[0]) + "_" + str(new_shape[1]) + ".h5", "w")
					create_save_flag = False

				if class_ == "":
					class_ = "data"

					# apply CLAHE for contrast enhancement
					if clahe_flag:
						clahe_create = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
						img = clahe_create.apply(img)
				else:
					img = minmaxscale(img.astype(np.float32), scale_=1).astype(np.uint8)

				if img.shape[1] < img.shape[0]:
					tmp = np.zeros((img_size, img_size), dtype=np.uint8)
					img_shapes = img.shape
					tmp[:img_shapes[0], :img_shapes[1]] = img
					img = tmp

				f.create_dataset(class_, data=img, compression="gzip", compression_opts=4)

			# finally close file, when finished writing to it
			f. close()


# preprocess the data
preprocess_segmentation_samples()








