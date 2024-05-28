# Deepfake Detection using CNN

This repository contains a deep learning model built with TensorFlow and Keras to detect whether an image is real or a deepfake (artificially generated). The model uses a Convolutional Neural Network (CNN) architecture to analyze and classify images.

## Features

- Convolutional Neural Network (CNN) architecture for image classification
- Trained on a dataset of real and deepfake images
- Streamlit web app for easy model deployment and inference
- Data augmentation techniques for improved model performance

## Installation

1. Clone the repository:
   
   git clone https://github.com/your-repo/deepfake-detection-cnn.git

2. Install the required dependencies:
   
   `pip install tensorflow keras numpy pillow streamlit`

## Dataset

The model was trained on a dataset containing real and deepfake images. Here is the link to download the dataset from Kaggle : https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images/data

## Usage

1. Prepare your dataset:
   - Organize your dataset into two folders: `train` and `val` (or `validation`).
   - Inside each folder, create subfolders for `real` and `fake` images.
   - Place your training images in the respective `real` and `fake` subfolders under the `train` folder.
   - Place your validation images in the respective `real` and `fake` subfolders under the `validation` folder.

2. Update the paths in the code:
   - Open `train_model.py` and replace the `train_dir` and `val_dir` paths with the appropriate paths to your `train` and `val` folders, respectively.

3. Train the model:
   - Run the `train_model.py` script to train the CNN model on your dataset.
   - The trained model will be saved as `best_model.h5` in the project directory.

4. Run the Streamlit app:
   - Execute the `predict.py` script to start the Streamlit web app.
   - Upload an image to the app, and it will predict whether the image is real or a deepfake.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.
