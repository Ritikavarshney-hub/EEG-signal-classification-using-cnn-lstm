 ## EEG Emotion Recognition using CNN-LSTM (DEAP Dataset)
 ## Overview

This project focuses on emotion recognition from EEG (electroencephalogram) signals using a hybrid deep learning architecture (CNN + LSTM).
We utilize the DEAP dataset, which contains EEG and peripheral physiological signals recorded while subjects watched emotion-eliciting videos.

The model learns spatio-temporal features from EEG signals to classify emotional states such as valence, arousal, and dominance.

## Key Features

Preprocessing and normalization of EEG signals from DEAP dataset.

Temporal feature extraction using 1D CNN layers.

Sequence modeling using LSTM layers.

Emotion classification (high/low valence & arousal).

Visualization of EEG signal patterns and model training results.

## Dataset Information

Dataset: DEAP - A Dataset for Emotion Analysis using Physiological Signals

Files: Each participant’s EEG data is stored in .dat or .mat format (32 channels, 40 trials).

Labels: Valence, Arousal, Dominance, and Liking (rated on a 1–9 scale).

## Model Architecture
| Layer            | Description                         | Output Shape              |
| ---------------- | ----------------------------------- | ------------------------- |
| Conv1D           | Feature extraction from EEG signals | (batch, seq_len, filters) |
| BatchNorm + ReLU | Normalization and activation        | -                         |
| LSTM             | Temporal modeling                   | (batch, hidden_dim)       |
| Dense            | Fully connected layers              | (batch, num_classes)      |
| Softmax          | Emotion probability distribution    | -                         |

## References

Koelstra, S., et al. (2012). DEAP: A database for emotion analysis using physiological signals.

https://www.kaggle.com/datasets/harshilgupta28/deap-dataset

Li, M., et al. (2019). EEG-based Emotion Recognition using CNN and LSTM Networks.
