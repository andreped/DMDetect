# DMDetect

Code relevant for training, evaluating, assessing and deploying CNN classifiers for Digital Mammography (DM) image classification.

This repository contains all necessary scripts to train and evaluate CNN image classifiers on a specific Kaggle dataset of DM images.
For this project we have used TensorFlow 2.4. This enabled us to experiment with TFRecords and tf.data.Dataset, which is suitable for efficient batch generation during training.

The data set used can be downloaded from [here](https://www.kaggle.com/skooch/ddsm-mammography/discussion/225969).

### Preliminary results

I've trained a CNN that detects images containing breast cancer tumour tissue. We get quite good results, without really tuning the network or training for long. A summary of the results can be seen below:

<!-- 
   Classes   |  Precision  |  Recall  |  F1-score  |  Support    
-------------|-------------|----------|------------|----------
           0 |    0.99     |   0.98   |    0.98    |   9755
           1 |    0.88     |   0.90   |    0.89    |   1445
-------------|-------------|----------|------------|----------
  Accuracy   |             |          |    0.97    |  11200
 macro avg   |    0.93     |   0.94   |    0.94    |  11200
weighted avg |    0.97     |   0.97   |    0.97    |  11200
--->

<img src="figures/performance_metrics.png" width="50%" height="50%">

Reaching a macro-average F1-score of 94% is a good start.

### Explainable AI (XAI)

To further assess the performance of the method, I used XAI to see if the method is doing what it should:

<img src="figures/XAI_example.png" width="50%" height="50%">

From this image, it seems like the model is reacting on the right part of the image. However, the network seems biased towards "always" using the central part of the image, at least as a default, if nothing else is found. This might be suboptimal. I will experiment with different data augmentation designs to assess whether can make the design more robust.


### How to use?

Given that you have: 1) Created a virtual environent, 2) installed all requirements, 3) defined the project as below, 4) updated the paths in train.py and eval.py, you should be ready to go.

Simply train a CNN classifier running the train.py script: 
```
python train.py
```

When a model is ready, it can be evaluated using the eval.py script, which will return summary performance results, as well as the option to further assessing the model using XAI.
```
python eval.py
```

### Project structure

```
+-- {DMDetect}/
|   +-- python/
|   |   +-- create_data.py
|   |   +-- train.py
|   |   +-- [...]
|   +-- data/
|   |   +-- folder_containing_the_unzipped_kaggle_dataset/
|   |   |   +-- fold_name0/
|   |   |   +-- fold_name1/
|   |   |   +-- [...]
|   +-- output/
|   |   +-- history/
|   |   |   +--- history_some_run_name1.txt
|   |   |   +--- history_some_run_name2.txt
|   |   |   +--- [...]
|   |   +-- models/
|   |   |   +--- model_some_run_name1.h5
|   |   |   +--- model_some_run_name2.h5
|   |   |   +--- [...]
```

### TODOs (most important from top to bottom):

- [x] Setup batch generation through TFRecords for GPU-accelerated generation and data augmentation
- [x] Introduce smart losses and metrics for handling class-imbalance 
- [x] Make end-to-end pipeline for automatic DM assessment
- [x] Achieve satisfactory classification performance
- [x] Introduce XAI-based method to further assess classifier
- [x] Test MTL design on the multi-classification tasks
- [x] Made proper support for MIL classifiers, that works both during training and inference 
- [x] Fix data augmentation scheme in the get_dataset method
- [ ] Find the optimal set of augmentation methods
- [ ] Get access to raw DM images, and test the pipeline across the full image (model trained on patches)
- [ ] Extract the distrbution between the 5 classes, to be used for balancing classes during training
- [ ] Introduce ROC-curves and AUC as additional metric for evaluating performance
- [ ] Make simple script for plotting losses and metrics as a function of epochs, using the CSV history

### Small tips

Make virtual environment:

> `virtualenv -ppython3 venv --clear`
`.\venv\Scripts\activate.ps1`
`pip install -r requirements.txt`

Activating virtual environment (on Win10):

> `.\venv\Scripts\activate.ps1`

Updating requirements.txt file:

> `pip freeze > requirements.txt`
