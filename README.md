# Anomaly Detection for Face Forgery

A zero-forgery-sample framework for detecting face forgeries using anomaly detection.

## Requirements

```bash
pip install -r requirements.txt
```

Required packages:
- opencv-python>=4.5.0
- numpy>=1.19.0
- scikit-learn>=0.24.0
- scikit-image>=0.18.0
- joblib>=1.0.0
- matplotlib>=3.3.0
- tqdm>=4.60.0

## Project Structure

```
.
├── config.py         # Configuration parameters
├── features.py       # Feature extraction module
├── model.py          # Model training and detection
├── utils.py          # Utility functions
├── train.py          # Training script
├── test.py           # Testing script
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Configuration

Edit `config.py` to set your dataset paths:

```python
# Dataset paths
BASE_PATH = '/path/to/your/dataset/real_images'
TRAIN_DATASETS = ['train_0']  # Training dataset names

# Real image datasets for testing
REAL_DATASETS = [
    '/path/to/your/real/images/dataset1',
    '/path/to/your/real/images/dataset2',
]

# Fake image datasets for testing
FAKE_DATASETS = [
    '/path/to/your/fake/images/dataset1',
    '/path/to/your/fake/images/dataset2',
]
```

## Dataset Format

- Images should be in common formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`
- Real images for training: Place in `BASE_PATH/train_0/`
- Test datasets: Organize real and fake images in separate folders

## Running the Code

### Step 1: Train the Model

```bash
python train.py
```

This will:
- Load real images from `BASE_PATH/train_0/`
- Train an Isolation Forest model
- Save the model as `model_train_0.pkl`
- Generate `training_times.csv` with training statistics

### Step 2: Test the Model

```bash
python test.py
```

This will:
- Load the trained model
- Test on all dataset pairs defined in `config.py`
- Process images using multi-threading for efficiency
- Generate detection results

## Output Files

After running the test script, you'll get:

1. **`detection_results_model_train_0.txt`** - Detailed results for each dataset pair:
   - Real/fake image accuracy
   - Performance metrics (accuracy, precision, recall, F1, AUC, AP)

2. **`summary_metrics_model_train_0.txt`** - Overall summary:
   - Total processing time and speed
   - Average metrics across all datasets
   - Overall performance metrics

## Performance Configuration

Adjust in `config.py` for your system:

```python
MAX_WORKERS = min(multiprocessing.cpu_count(), 8)  # Number of threads
BATCH_SIZE = MAX_WORKERS * 4  # Batch size for processing
```

## Example Usage

### Training with Custom Dataset

1. Create a folder structure:
   ```
   /your/dataset/path/
   └── train_0/
       ├── real_image_001.jpg
       ├── real_image_002.jpg
       └── ...
   ```

2. Update `config.py`:
   ```python
   BASE_PATH = '/your/dataset/path'
   ```

3. Run training:
   ```bash
   python train.py
   ```

### Testing on Your Data

1. Organize test images:
   ```
   /test/real/dataset1/
   ├── image1.jpg
   └── ...
   
   /test/fake/dataset1/
   ├── fake1.jpg
   └── ...
   ```

2. Update `config.py`:
   ```python
   REAL_DATASETS = ['/test/real/dataset1/']
   FAKE_DATASETS = ['/test/fake/dataset1/']
   ```

3. Run testing:
   ```bash
   python test.py
   ```

## Notes

- The model is trained only on real images (zero-forgery-sample approach)
- Detection threshold is set to identify 1% contamination in the training data
- Supports parallel processing for faster inference
- All metrics are saved automatically for analysis
