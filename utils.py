"""
Utility module for data handling and image processing
"""
import os
import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)


def load_real_images(path):
    """
    Load real images from a directory

    Args:
        path: Directory path containing images

    Returns:
        List of loaded images
    """
    real_images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            real_images.append(img)
    return real_images


def get_image_paths(dataset_path):
    """
    Get all image paths from a dataset directory

    Args:
        dataset_path: Path to the dataset directory

    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_paths = []

    try:
        for filename in os.listdir(dataset_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(dataset_path, filename))
    except Exception as e:
        print(f"Error accessing dataset {dataset_path}: {e}")

    return image_paths


def calculate_metrics(y_true, y_pred, y_proba):
    """
    Calculate various classification metrics

    Args:
        y_true: True labels (1 for real, -1 for fake)
        y_pred: Predicted labels
        y_proba: Prediction probabilities

    Returns:
        Dictionary of metrics
    """
    # Convert labels: 1 (real) -> 1, -1 (fake) -> 0 for sklearn compatibility
    y_true_binary = np.array([1 if label == 1 else 0 for label in y_true])
    y_pred_binary = np.array([1 if pred == 1 else 0 for pred in y_pred])

    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
    metrics['precision'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    metrics['recall'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    metrics['f1'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    # AUC (Area Under ROC Curve)
    try:
        metrics['auc'] = roc_auc_score(y_true_binary, y_proba)
    except ValueError as e:
        print(f"Warning: Could not calculate AUC: {e}")
        metrics['auc'] = 0.0

    # AP (Average Precision)
    try:
        metrics['ap'] = average_precision_score(y_true_binary, y_proba)
    except ValueError as e:
        print(f"Warning: Could not calculate AP: {e}")
        metrics['ap'] = 0.0

    return metrics