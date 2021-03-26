import numpy as np
import os
from create_data import get_dataset
from tensorflow.keras.models import load_model
from tqdm import tqdm
from sklearn.metrics import classification_report, auc, roc_curve
import matplotlib.pyplot as plt 


# @TODO: perhaps just use .ravel() instead?
def flatten_(tmp):
	out = []
	for t in tmp:
		for t2 in t:
			out.append(t2)
	out = np.array(out)
	return out


# whether or not to use GPU for training (-1 == no GPU, else GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# paths
data_path = "C:/Users/andrp/workspace/DeepXDMDetect/data/DDSM_mammography_data/"
save_path = "C:/Users/andrp/workspace/DeepXDMDetect/output/models/"
history_path = "C:/Users/andrp/workspace/DeepXDMDetect/output/history/"

name = "260321_212637_classifier_model"

BATCH_SIZE = 32
num_classes = 2
SHUFFLE_FLAG = False
instance_size = (50, 50, 1)
N_SAMPLES = 55890
# th = 0.5  # threshold to use for binarizing prediction

# get independent test set to evaluate trained model
test_set = get_dataset(BATCH_SIZE, data_path + "training10_4/", num_classes, SHUFFLE_FLAG, instance_size[:-1])

# load trained model (for deployment or usage in diagnostics - freezed, thus deterministic, at least in theory)
model = load_model(save_path + name + ".h5", compile=False)

preds = []
preds_conf = []
gts = []
for cnt, (x_curr, y_curr) in tqdm(enumerate(test_set), total=int(N_SAMPLES / 5 / BATCH_SIZE)):
	gts.append(np.argmax(y_curr, axis=1))
	pred_conf = model.predict(x_curr)
	pred_final = np.argmax(pred_conf, axis=1)  # using argmax for two classes here is equivalent with using th=0.5. Essentially, chooses the most confidence class as the predicted class
	preds.append(pred_final)

	if cnt == int(N_SAMPLES / 5 / BATCH_SIZE):
		break

preds = flatten_(preds).astype(np.int32)
gts = flatten_(gts).astype(np.int32)

# get summary statistics (performance metrics)
summary = classification_report(gts, preds)
print(summary)

# @TODO: Plot ROC and report AUC as additional performance metric(s)




