# Image-Compression-with-Neural-Networks

# Weights for PU-PieApp metric
Weights that are used by PU-PieApp metric can be downloaded from here: https://github.com/gfxdisp/pu_pieapp/releases/download/v0.0.1/pupieapp_weights.pt

# Using PU-PieApp metric
Metric can be used in two ways:
1. Running compute_metric_values notebook to compute metric values for pairs of original and reconstructed images from original_and_reconstructed_images directory (create it in root project location if it doesn't exist). Metric won't be used as a part of ML model then but will evaluate compression effectiveness instead.
2. Using it together with models - metric needs to be imported and attached to compiled model: (model.compile(optimizer=opt, loss='mse', run_eagerly=True, metrics=[PUPieAppMetric()]))