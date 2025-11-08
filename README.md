# ECG-signal-classification-using-cnn-lstm
## Project Overview

This project implements an ECG Signal Classification System using a hybrid CNN-BiLSTM architecture in PyTorch to automatically detect cardiac arrhythmias from the MIT-BIH Arrhythmia Dataset.
By combining convolutional neural networks (CNNs) for spatial feature extraction and Bidirectional LSTMs for temporal pattern recognition, the model effectively learns both waveform shapes and heartbeat sequences, achieving robust classification performance.

## Objectives

Detect heart arrhythmias from ECG signals automatically.

Develop a hybrid CNN-BiLSTM architecture to capture both morphological and temporal ECG patterns.

Address data imbalance (83% normal beats) through class weighting.

Evaluate model performance using confusion matrix and ROC curves.

## Theoretical Background

CNN (Convolutional Neural Network) captures spatial features (P-wave, QRS complex, T-wave morphology).

BiLSTM (Bidirectional LSTM) captures temporal dependencies — how beats evolve over time.

Hybrid CNN-BiLSTM combines spatial and sequential learning for robust ECG signal understanding.

## Dataset

Source: MIT-BIH Arrhythmia Database (PhysioNet)

Classes (Example 5 types):

Normal (N)

Atrial Premature Beat (A)

Ventricular Premature Beat (V)

Left Bundle Branch Block (L)

Right Bundle Branch Block (R)

Preprocessing Steps:

Extracted ECG segments centered around R-peaks.

Normalized signals using MinMax scaling.

Applied bandpass filtering to remove noise.

Addressed class imbalance with weighted loss.

## Model Architecture
1️⃣ CNN Feature Extractor

3 Convolutional blocks (Conv1D + BatchNorm + ReLU + MaxPool)

Extracts morphological features from ECG signals.

2️⃣ BiLSTM Temporal Model

2 Bidirectional LSTM layers to learn sequential dependencies.

3️⃣ Classifier Head

Fully connected layers with dropout for regularization.

Softmax activation for multi-class classification.

## Training Setup

Framework: PyTorch

Optimizer: Adam

Loss Function: CrossEntropyLoss with class weights

Learning Rate Scheduling: StepLR

Epochs: 50

Batch Size: 64

## Evaluation Metrics: Accuracy, F1-score, ROC-AUC, Confusion Matrix

Results
Metric	Value
Accuracy	91.4%
F1-score	0.89
ROC-AUC	0.93

Visualizations:

Confusion Matrix → Class-wise performance

ROC Curves → Multi-class discrimination ability

Sample ECG plots → Signal morphology understanding


##  Key Learnings

Practical understanding of deep learning for biosignals.

Experience with CNNs for feature extraction and LSTMs for sequence modeling.

Gained skills in data preprocessing, imbalanced classification, and evaluation metrics.

Hands-on exposure to PyTorch, signal visualization, and model interpretability.

## Future Extensions

Deploy the model as a real-time ECG monitoring system using Flask or Streamlit.

Integrate with wearable IoT sensors for live arrhythmia detection.

Explore transformer-based time-series models (e.g., Time Series Transformer).

Extend to multi-lead ECG analysis for richer feature extraction.

Implement explainable AI (XAI) methods for medical interpretability.

## References

Moody, G.B., Mark, R.G. “MIT-BIH Arrhythmia Database.” PhysioNet.

Kiranyaz, S. et al. “Real-Time Patient-Specific ECG Classification by 1-D Convolutional Neural Networks.” IEEE Trans. Biomed. Eng., 2015.

PyTorch Documentation: https://pytorch.org/docs/stable/
