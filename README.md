# 🧠 EEG Classification using Machine Learning

This project applies signal processing and machine learning techniques to classify EEG signals related to motor activity.

## Overview

The system processes real EEG data and identifies patterns associated with different motor states.

## Steps

- Signal filtering (8–30 Hz)
- Epoch segmentation
- Channel selection (C3, C4, Cz)
- Feature extraction (log-variance)
- Classification using SVM

## Results

Achieved approximately **70% accuracy** in classifying motor-related EEG signals.

## Technologies

- Python
- MNE
- NumPy
- Scikit-learn
- Matplotlib

## Motivation

This project was developed to explore applications of data analysis and machine learning in biomedical engineering and neurotechnology.

---

Future improvements:
- Use multiple subjects
- Improve feature extraction
- Build interactive dashboard
