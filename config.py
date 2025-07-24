"""
Configuration file for all parameters
"""
import multiprocessing

# Model parameters
KERNEL_SIZES = [10, 15, 20]  # Kernel sizes for LMD-DCT feature extraction
N_FEATURES_TO_SELECT = 9  # Number of features to select

# Performance parameters
MAX_WORKERS = min(multiprocessing.cpu_count(), 8)  # Maximum number of threads
BATCH_SIZE = MAX_WORKERS * 4  # Batch size for processing

# Dataset paths
BASE_PATH = '/path/to/your/dataset/real_images'
TRAIN_DATASETS = ['train_0']

# Real image datasets
REAL_DATASETS = [
    '/path/to/your/real/images/dataset1',

]

# Fake image datasets
FAKE_DATASETS = [
    '/path/to/your/fake/images/dataset1',

]

# Model filenames
MODEL_FILENAMES = ['model_train_0.pkl']