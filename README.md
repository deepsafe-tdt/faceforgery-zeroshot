# Anomaly Detection for Face Forgery: A Zero-Forgery-Sample Framework Using Multi-Scale Dark Channel DCT Features

## Overview

This repository implements a novel approach for detecting face forgeries using anomaly detection techniques. The framework operates on a **zero-forgery-sample** principle, meaning it is trained exclusively on authentic images and identifies forgeries as anomalies without requiring fake samples during training.

## Key Features

- **Zero-Shot Learning**: Trained only on real images, no fake samples needed
- **Multi-Scale Feature Extraction**: Combines LMD-DCT, HSV histogram, and LBP features
- **Efficient Processing**: Multi-threaded batch processing for high-speed detection
- **Robust Detection**: Uses Isolation Forest for anomaly detection
- **Comprehensive Evaluation**: Includes metrics like accuracy, precision, recall, F1-score, AUC, and AP

## Methodology

### Feature Extraction
The system extracts three types of features:

1. **LMD-DCT Features**: Local Mean Decomposition with Discrete Cosine Transform using multiple kernel sizes (15, 30, 45)
2. **HSV Histogram Features**: Color distribution analysis in HSV color space
3. **LBP Histogram Features**: Local Binary Pattern for texture analysis

### Anomaly Detection
- Uses Isolation Forest algorithm trained exclusively on authentic face images
- Identifies forgeries as anomalies that deviate from the learned authentic face distribution

## Requirements