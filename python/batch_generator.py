import numpy as np
import os
import hdf5


# This function is probably not needed with the Kaggle data set, as TFRecords are used, and then TF-Keras already has a way of generating batches during training