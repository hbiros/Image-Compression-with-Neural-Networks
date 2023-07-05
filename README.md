# Image-Compression-with-Neural-Networks
## Usage
Clone the repository:
```
git clone https://github.com/hbiros/Image-Compression-with-Neural-Networks.git
```
Dependencies can be installed by running:
```
pip install -r requirements.txt
```
## Training the models
```
python src/train.py 
```
Options:
**-m**, **--model** - Specify the name of the model you want to train (models are placed in the src/models/ package). <br />
**-e**, **--epochs** - Numer of epochs. <br />
**-bs**, **--batch_size** - Batch size. <br />
**-s**, **--save** -  Save the model after training (the model will be saved in a folder named according to the date and time of the start of training). <br />
**-t**, **--train_data** - Specify data to train the model. The training dataset should be a numpy binary array of 64x64 RGB image fragments. If you don't want to use your own dataset, use src/resources/get_images.py to download and prepare the data. <br />
**-r**, **--ratio** - Float between 0 and 1. Fraction of the training data to be used as validation data.  <br />

## Using the model
```
python src/main.py
```
Options:
**-m**, **--model** - Name of the folder containing the model parameters.
**-i**, **--img_name** - Name of image to be processed by compression. The reconstructed image will be saved under the original name with the suffix "_reconstructed".
## Weights for PU-PieApp metric
Weights that are used by PU-PieApp metric can be downloaded from here: https://github.com/gfxdisp/pu_pieapp/releases/download/v0.0.1/pupieapp_weights.pt

## Using PU-PieApp metric
Metric can be used in two ways:
1. Running compute_metric_values notebook to compute metric values for pairs of original and reconstructed images from original_and_reconstructed_images directory (create it in root project location if it doesn't exist). Metric won't be used as a part of ML model then but will evaluate compression effectiveness instead.
2. Using it together with models - metric needs to be imported and attached to compiled model: (model.compile(optimizer=opt, loss='mse', run_eagerly=True, metrics=[PUPieAppMetric()]))
