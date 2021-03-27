# DeepXDMDetect
Code relevant for training, evaluating, assessing and deploying CNN classifiers for DM image classification

This code as of now, contains all necessary scripts to train and evaluate CNN image classifiers on a specific Kaggle dataset of DM images.
For this project we have used TensorFlow 2.4. This has made it possible for us to use TFRecords, which are suitable for simple batch generation during training and inference. It is also extremely efficient.

### Preliminary results
I've trained a CNN that detects images containing breast cancer tumour tissue. We get quite good results, without really tuning the network or training for long. A summary of the results can be seen below:
![Alt text](figures/performance_metrics.png)

Reaching a macro-average F1 of 94% is a good start. 

### Explainable AI (XAI)
To further assess the performance of the method, I used XAI to see if the method is doing what it should:

![Alt text](figures/XAI_example.png)

From this image, it seems like the model is reacting on the right part of the image. However, the network seems biased towards "always" using the central part of the image, at least as a default, if nothing else is found. This might be suboptimal. I will introduce some shift augmentation tomorrow to see if it helps.

### Small tips

Make virtual environment:

> `virtualenv -ppython3 venv --clear`
`.\venv\Scripts\activate.ps1`
`pip install -r requirements.txt`

Activating virtual environment (on Win10):

> `.\venv\Scripts\activate.ps1`

Updating requirements.txt file:

> `pip freeze > requirements.txt`
