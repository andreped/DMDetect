# DeepXDMDetect
Code relevant for training, evaluating, assessing and deploying CNN classifiers for DM image classification

This code as of now, contains all necessary scripts to train and evaluate CNN image classifiers on a specific Kaggle dataset of DM images.
For this project we have used TensorFlow 2.4. This has made it possible for us to use TFRecords, which are suitable for simple batch generation during training and inference. It is also extremely efficient.

### Small tips

Make virtual environment:

> `virtualenv -ppython3 venv --clear`
`.\venv\Scripts\activate.ps1`
`pip install -r requirements.txt`

Activating virtual environment (on Win10):

> `.\venv\Scripts\activate.ps1`

Updating requirements.txt file:

> `pip freeze > requirements.txt`
