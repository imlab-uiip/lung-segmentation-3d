# Lung Segmentation (3D)
Repository features [UNet](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) inspired architecture used for segmenting lungs on chest 3D tomography images.

## Demo
Run `inference.py` to see the application of the model on [Demo](https://github.com/imlab-uiip/lung-segmentation-3d/tree/master/Demo) files.

## Implementation
Implemented in Keras(2.0.4) with TensorFlow(1.1.0) as backend. 

To use this implementation one needs to load and preprocess data (see `load_data.py`), train new model if needed (`train_model.py`) and use the model for generating lung masks (`inference.py`).

`trained_model.hdf5` and `trained_model_wc.hdf5` contain models trained on private data set without and with coordinates channels.

## Segmentation
![](https://github.com/imlab-uiip/lung-segmentation-3d/blob/master/Demo/Predictions/id003-128x128x64.nii.gz-preview.png)
![](https://github.com/imlab-uiip/lung-segmentation-3d/blob/master/Demo/Predictions/id002-128x128x64.nii.gz-preview.png)
