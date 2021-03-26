import numpy as np
import os
from create_data import get_dataset
from tensorflow.keras.models import load_model
from tqdm import tqdm
from sklearn.metrics import classification_report, auc, roc_curve
import matplotlib.pyplot as plt
from tf_explain.core import GradCAM
import tensorflow as tf
from utils import flatten_


# turn off eager execution
#tf.compat.v1.disable_eager_execution()


# whether or not to use GPU for training (-1 == no GPU, else GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # set to 0 to use GPU

# paths
data_path = "C:/Users/andrp/workspace/DeepXDMDetect/data/DDSM_mammography_data/"
save_path = "C:/Users/andrp/workspace/DeepXDMDetect/output/models/"
history_path = "C:/Users/andrp/workspace/DeepXDMDetect/output/history/"

name = "260321_234024_classifier_model"

BATCH_SIZE = 64
num_classes = 2
SHUFFLE_FLAG = False
instance_size = (160, 160, 3)
N_SAMPLES = 55890
XAI_FLAG = False  # False
# th = 0.5  # threshold to use for binarizing prediction

# get independent test set to evaluate trained model
test_set = get_dataset(BATCH_SIZE, data_path + "training10_4/*", num_classes, SHUFFLE_FLAG, instance_size[:-1], train_mode=False)

# load trained model (for deployment or usage in diagnostics - freezed, thus deterministic, at least in theory)
model = load_model(save_path + name + ".h5", compile=False)

preds = []
preds_conf = []
gts = []
for cnt, (x_curr, y_curr) in tqdm(enumerate(test_set), total=int(N_SAMPLES / 5 / BATCH_SIZE)):
	gt_class = np.argmax(y_curr, axis=1)
	gts.append(gt_class)
	pred_conf = model.predict(x_curr)
	pred_final = np.argmax(pred_conf, axis=1)  # using argmax for two classes here is equivalent with using th=0.5. Essentially, chooses the most confidence class as the predicted class
	preds.append(pred_final)

	# if XAI_FLAG is enabled, we will use Explainable AI (XAI) to assess if a CNN is doing what it should (what is it using in the image to solve the task)
	# NOTE: Will only display first element in batch
	if XAI_FLAG:
		img = tf.keras.preprocessing.image.img_to_array(x_curr[0])
		data = ([img], None)

		explainer = GradCAM()
		grid = explainer.explain(data, model, class_index=pred_final[0])

		fig, ax = plt.subplots(1, 2)
		ax[0].imshow(img, cmap="gray")
		ax[1].imshow(grid, cmap="gray")
		ax[1].set_title("Pred: " + str(pred_final[0]) + ", GT: " + str(gt_class[0]))
		plt.show()

	if cnt == int(N_SAMPLES / 5 / BATCH_SIZE):
		break

preds = flatten_(preds).astype(np.int32)
gts = flatten_(gts).astype(np.int32)

# get summary statistics (performance metrics)
summary = classification_report(gts, preds)
print(summary)

# @TODO: Plot ROC and report AUC as additional performance metric(s)




