"""
Model training and evaluation module
"""
import cv2
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import multiprocessing
from features import compute_all_features


def train_model(real_images, kernel_sizes, n_features_to_select):
    """
    Train an Isolation Forest model for anomaly detection

    Args:
        real_images: List of real images for training
        kernel_sizes: Kernel sizes for feature extraction
        n_features_to_select: Number of features to select

    Returns:
        Tuple of (model, scaler, selector)
    """
    # Extract features from all images
    features = []
    for img in real_images:
        features.append(compute_all_features(img, kernel_sizes))

    features = np.array(features)

    # Standardize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Feature selection using SelectKBest
    selector = SelectKBest(score_func=f_classif, k=n_features_to_select)
    selector.fit(features, np.ones(features.shape[0]))
    selected_features = selector.transform(features)

    # Train Isolation Forest model
    model = IsolationForest(
        n_estimators=200,
        max_samples='auto',
        contamination=0.01,
        random_state=42
    )
    model.fit(selected_features)

    return model, scaler, selector


def detect_image(image_path, model, scaler, selector, kernel_sizes, return_proba=False):
    """
    Detect if an image is real or fake

    Args:
        image_path: Path to the image
        model: Trained model
        scaler: Feature scaler
        selector: Feature selector
        kernel_sizes: Kernel sizes for feature extraction
        return_proba: Whether to return probability

    Returns:
        Prediction (and probability if requested)
    """
    # Load image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Warning: Could not load image at {image_path}")
        if return_proba:
            return -1, 0.0
        return -1

    # Extract and transform features
    combined_features = compute_all_features(image, kernel_sizes)
    combined_features = scaler.transform([combined_features])
    selected_features = selector.transform(combined_features)

    # Make prediction
    prediction = model.predict(selected_features)[0]

    if return_proba:
        # Get prediction probability
        try:
            if hasattr(model, 'decision_function'):
                # Use decision function and apply sigmoid
                decision = model.decision_function(selected_features)[0]
                prob_real = 1 / (1 + np.exp(-decision))
            else:
                # Fallback: use prediction as probability
                prob_real = 1.0 if prediction == 1 else 0.0
        except Exception as e:
            print(f"Warning: Could not get probability for {image_path}: {e}")
            prob_real = 1.0 if prediction == 1 else 0.0

        return prediction, prob_real

    return prediction


def process_single_image(args):
    """
    Process a single image for parallel processing

    Args:
        args: Tuple of (image_path, model, scaler, selector, kernel_sizes)

    Returns:
        Tuple of (path, prediction, probability, success)
    """
    image_path, model, scaler, selector, kernel_sizes = args
    try:
        prediction, probability = detect_image(
            image_path, model, scaler, selector, kernel_sizes, return_proba=True
        )
        return image_path, prediction, probability, True
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return image_path, -1, 0.0, False


def batch_detect_images(image_paths, model, scaler, selector, kernel_sizes,
                        max_workers=None, batch_size=None):
    """
    Batch detect images using multi-threading

    Args:
        image_paths: List of image paths
        model: Trained model
        scaler: Feature scaler
        selector: Feature selector
        kernel_sizes: Kernel sizes for feature extraction
        max_workers: Maximum number of worker threads
        batch_size: Size of each batch

    Returns:
        Tuple of (paths, predictions, probabilities)
    """
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)

    if batch_size is None:
        batch_size = max_workers * 4

    all_predictions = []
    all_probabilities = []
    all_paths = []
    successful_count = 0

    print(f"Processing {len(image_paths)} images using {max_workers} threads...")

    # Process in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_args = [(path, model, scaler, selector, kernel_sizes) for path in batch_paths]

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_desc = f"Batch {i // batch_size + 1}/{(len(image_paths) - 1) // batch_size + 1}"
            results = list(tqdm(
                executor.map(process_single_image, batch_args),
                total=len(batch_args),
                desc=batch_desc,
                leave=False
            ))

        # Collect results
        for path, prediction, probability, success in results:
            if success:
                all_predictions.append(prediction)
                all_probabilities.append(probability)
                all_paths.append(path)
                successful_count += 1

    print(f"Successfully processed {successful_count}/{len(image_paths)} images")
    return all_paths, all_predictions, all_probabilities