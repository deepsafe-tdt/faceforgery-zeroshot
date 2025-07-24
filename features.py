"""
Feature extraction module
Contains all image feature extraction functions including LMD-DCT, HSV histogram, and LBP histogram
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def extract_enhanced_dct_features(image, kernel_sizes=[15, 30, 45]):
    """
    Extract enhanced DCT features using LMD (Local Mean Decomposition) and DCT

    Args:
        image: Input image
        kernel_sizes: List of kernel sizes for dark channel computation

    Returns:
        Array of DCT features
    """
    all_features = []

    # Get DCT features for each kernel size
    for kernel_size in kernel_sizes:
        # Convert to dark channel (part of LMD process)
        dark_channel = compute_dark_channel(image, kernel_size)

        # Apply DCT transformation
        dct = cv2.dct(np.float32(dark_channel))
        magnitude_spectrum = np.abs(dct)
        rows, cols = dark_channel.shape

        # Extract energy of different frequency regions
        kernel_features = []
        for i in range(1, 80):
            freq_region = magnitude_spectrum[:rows // i, :cols // i]
            kernel_features.append(np.sum(freq_region))

        all_features.extend(kernel_features)

    return np.array(all_features)


def compute_hsv_features(img):
    """
    Compute HSV histogram features

    Args:
        img: Input image in BGR format

    Returns:
        Flattened HSV histogram features
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Calculate histograms for each channel
    hist_h = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])

    # Concatenate and flatten
    return np.concatenate((hist_h, hist_s, hist_v)).flatten()


def compute_dark_channel(img, kernel_size):
    """
    Compute dark channel for LMD-DCT feature extraction

    Args:
        img: Input image
        kernel_size: Size of the morphological kernel

    Returns:
        Inverted dark channel
    """
    # Convert RGB to LAB color space for better color representation
    lab_image = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # Extract L, A, B channels
    l, a, b = cv2.split(lab_image)

    # Find minimum value across all three channels
    dark = cv2.min(cv2.min(l, a), b)

    # Apply morphological erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dark_channel = cv2.erode(dark, kernel)

    # Return inverted dark channel
    return 1 - dark_channel


def compute_lbp_features(img):
    """
    Compute Local Binary Pattern (LBP) histogram features

    Args:
        img: Input image in BGR format

    Returns:
        LBP histogram features
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute LBP with 8 neighbors and radius 1
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')

    # Calculate histogram
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))

    return hist


def compute_all_features(img, kernel_sizes):
    """
    Compute all features: LMD-DCT, HSV histogram, and LBP histogram

    Args:
        img: Input image
        kernel_sizes: List of kernel sizes for LMD-DCT

    Returns:
        Concatenated feature vector
    """
    # Extract LMD-DCT features
    dct_features = extract_enhanced_dct_features(img, kernel_sizes)

    # Extract HSV histogram features
    hsv_features = compute_hsv_features(img)

    # Extract LBP histogram features
    lbp_features = compute_lbp_features(img)

    # Concatenate all features
    return np.concatenate((dct_features, hsv_features, lbp_features))