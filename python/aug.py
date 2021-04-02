import numpy as np
import cv2


# vertical flip
def add_flip_vert(input_im, output):
    input_im = np.flip(input_im, 0)
    output = np.flip(output, 0)
    return input_im, output


# horizontal flip
def add_flip_horz(input_im, output):
    input_im = np.flip(input_im, 1)
    output = np.flip(output, 1)
    return input_im, output


# lossless rot90 
def add_rotation_ll(input_im, output):
    k = random_integers(1, high=3)  # randomly choose rotation angle: +-90, +,180, +-270

    # rotate
    input_im = np.rot90(input_im, k)
    output = np.rot90(output, k)

    return input_im, output


# gamma transform
def add_gamma(input_im, output, r_limits):
    r_min, r_max = r_limits

    # randomly choose gamma factor
    r = np.random.uniform(r_min, r_max)

    # apply transform
    input_im = np.clip(np.round(input_im ** r), a_min=0, a_max=255)

    # need to normalize again after augmentation
    input_im = input_im / 255

    return input_im, output


def augment_numpy(x, y, aug):
	# only apply aug if "aug" is not empty
	if not bool(aug):
		if 'vert' in aug:
			if (np.random.random_integers(0, 1) == 1):
				x, y = add_flip_vert(x, y)

		if 'horz' in aug:
			if (np.random.random_integers(0, 1) == 1):
				x, y = add_flip_horz(x, y)

		if 'rot90' in aug:
			if (np.random.random_integers(0, 1) == 1):
				x, y = add_rotation_ll(x, y)

		if 'gamma' in aug:
			if (np.random.random_integers(0, 1) == 1):
				x, y = add_gamma(x, y, aug["gamma"])

	return x, y