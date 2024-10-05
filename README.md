# Fault Diagnosis in Rotating Machinery Using Deep Learning
This repository contains code and documentation for a project on diagnosing mechanical faults in rotating machinery using deep learning techniques. The project focuses on classifying different fault conditions based on vibration signal data, leveraging both time-domain and frequency-domain analyses.
## Introduction
Mechanical faults in rotating machinery, such as misalignment, unbalance, and looseness, can lead to significant operational issues if not detected early. This project aims to develop a deep learning model capable of classifying these fault conditions based on vibration signal data collected from machinery.

By comparing models trained on time-domain data and frequency-domain data (obtained via Fast Fourier Transform), the project explores the effectiveness of different data representations in fault diagnosis.

## Dataset Citation

This project uses the dataset from the following paper:

Brito, L.C., Susto, G.A., Brito, J.N., & Duarte, M.A.V. (2023). *Fault Diagnosis using eXplainable AI: A transfer learning-based approach for rotating machinery exploiting augmented synthetic data*. Expert Systems with Applications, 232(4), 120860. DOI: [10.1016/j.eswa.2023.120860](https://doi.org/10.1016/j.eswa.2023.120860)

The dataset consists of vibration signal data collected under four different conditions:

Normal Condition
Misalignment
Unbalance
Looseness
Each condition includes multiple samples of multichannel time-series data. The data files are stored in NumPy .npy format.

## Project Structure
├── data/
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_test.npy
├── models/
│   ├── cnn1d_model.py
│   ├── fault_classifier_time.pth
│   ├── fault_classifier_fft.pth
├── scripts/
│   ├── create_train_test_data.py
│   ├── training_time_model.py
│   ├── training_fft_model.py
│   └── utils.py
├── README.md
├── requirements.txt


### Directory Descriptions
- `data/`: Contains the training and testing datasets.
- `models/`: Contains the model definition and saved model weights.
- `scripts/`: Contains scripts for data preparation and model training.
- `README.md`: Project documentation.
- `requirements.txt`: List of required Python packages.


## Data Preparation
The create_train_test_data.py script loads the raw data from the dataset directory, assigns labels based on the fault condition, and splits the data into training and testing sets with stratification.

## Time-Domain Data
Normalization: The time-domain data is normalized to have zero mean and unit variance.
Data Shape: The input data has the shape (num_samples, channels, time_steps).
## Frequency-Domain Data
FFT Transformation: The time-domain data is transformed into the frequency domain using the Fast Fourier Transform (FFT).
Log Scaling: The magnitude of the FFT is scaled using a logarithmic function to compress the dynamic range.
Normalization: The frequency-domain data is normalized using the mean and standard deviation from the training data.
Data Shape: The input data has the shape (num_samples, channels, freq_bins).

## Model Architecture
The model used is a one-dimensional Convolutional Neural Network (1D CNN) defined in cnn1d_model.py. The architecture includes:

Convolutional Layers: Extract features from the input signals.
Pooling Layers: Reduce dimensionality and capture dominant features.
Global Average Pooling: Reduces the output from convolutional layers to a fixed size.
Fully Connected Layer: Outputs the class probabilities.
Training
Time-Domain Model
Script: training_time_model.py
Data: Raw time-domain data.
Epochs: 20
Optimizer: Adam optimizer with a learning rate of 0.001.
Loss Function: Cross-Entropy Loss.
Frequency-Domain Model
Script: training_fft_model.py
Data: Frequency-domain data obtained via FFT.
Epochs: 20
Optimizer: Adam optimizer with a learning rate of 0.001.
Loss Function: Cross-Entropy Loss.

## Training Procedure
Data Loading: The scripts load the preprocessed data from .npy files.
Model Initialization: The model is instantiated and moved to the available device (CPU or GPU).
Training Loop: The model is trained over the specified number of epochs, updating weights based on the loss computed.
Validation: After training, the model is evaluated on the test set to assess its performance.

## Results
Time-Domain Model Accuracy: Achieved an accuracy of 86% on the test data.
Frequency-Domain Model Accuracy: Achieved an accuracy of 96% on the test data.
The significant improvement in accuracy when using frequency-domain data suggests that the FFT transformation effectively highlights features relevant to fault diagnosis.
