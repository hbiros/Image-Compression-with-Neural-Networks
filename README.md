# Image-Compression-with-Neural-Networks
This repository contains Convolutional Neural Network (CNN) models for image compression using the autoencoder architecture. The models are designed to reduce the size of images while preserving their visual quality. Four different models are provided, each offering varying compression ratios:

1. **model_1** - 6x Compression Ratio
2. **model_2** - 1.5x Compression Ratio
3. **model_3** - 6x Compression Ratio
4. **model_4** - 24x Compression Ratio

Please note that all models process the images in 64x64 pixel fragments. Larger images will be split into multiple fragments for processing, and the resulting compressed image will be reassembled accordingly.

## Usage
Clone the repository:
```
git clone https://github.com/hbiros/Image-Compression-with-Neural-Networks.git
```
Dependencies can be installed by running:
```
pip install -r requirements.txt
```
## Data
For the training purposes, we provide a dataset of about 100,000 image fragments with a resolution of 64x64 pixels. To download and prepare the data, run:
```
python src/resources/get_images.py
```
Note that by default the data will be stored in a numpy binary file in the **data** folder.

## Training the models
```
python src/train.py 
```
Options:<br />
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
This script takes model parameters and an image path as arguments.

The input image is processed by splitting it into non-overlapping 64x64 pixel fragments. Each fragment is individually passed through the loaded models for compression, resulting in a compressed representation of the fragment.

After the compression process, the compressed fragments are reconstructed by decoding them using the decoder portion of the autoencoder models. The reconstructed fragments are then reassembled to produce the final reconstructed image. The reconstructed image will be saved under the original name with the suffix "_reconstructed".

To ensure proper alignment with the 64x64 pixel grid, the final reconstructed image may be clipped to a size that is a multiple of 64x64 pixels.

Options:<br />
**-m**, **--model** - Name of the folder containing the model parameters. <br />
**-i**, **--img_name** - Name of image to be processed by compression. <br />
## Weights for PU-PieApp metric
Weights that are used by PU-PieApp metric can be downloaded from here: https://github.com/gfxdisp/pu_pieapp/releases/download/v0.0.1/pupieapp_weights.pt

## Using PU-PieApp metric
Metric can be used in two ways:
1. Running compute_metric_values notebook to compute metric values for pairs of original and reconstructed images from original_and_reconstructed_images directory (create it in root project location if it doesn't exist). Metric won't be used as a part of ML model then but will evaluate compression effectiveness instead.
2. Using it together with models - metric needs to be imported and attached to compiled model: (model.compile(optimizer=opt, loss='mse', run_eagerly=True, metrics=[PUPieAppMetric()]))
