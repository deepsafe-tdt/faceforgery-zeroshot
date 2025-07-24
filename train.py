"""
Training script for image authenticity detection model
"""
import os
import joblib
import time
import csv
import warnings
from config import *
from utils import load_real_images
from model import train_model

# Ignore specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")


def main():
    """Main training function"""
    dataset_sizes = []
    training_times = []

    # Create CSV file to save training times
    with open('training_times.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Dataset', 'Size', 'Training Time (s)'])

        for dataset in TRAIN_DATASETS:
            print(f"Loading dataset: {dataset}")
            dataset_path = os.path.join(BASE_PATH, dataset)

            # Load real images
            real_images = load_real_images(dataset_path)
            dataset_size = len(real_images)
            dataset_sizes.append(dataset_size)
            print(f"Loaded {dataset_size} images")

            # Train model
            print("Training model...")
            start_time = time.time()
            model, scaler, selector = train_model(
                real_images, KERNEL_SIZES, N_FEATURES_TO_SELECT
            )
            end_time = time.time()

            training_time = end_time - start_time
            training_times.append(training_time)

            # Save model
            model_filename = f'model_{dataset}.pkl'
            joblib.dump((model, scaler, selector), model_filename)
            print(f"Model for {dataset} trained and saved to {model_filename}")
            print(f"Training time: {training_time:.2f} seconds")

            # Write to CSV
            csvwriter.writerow([dataset, dataset_size, training_time])

    print("\nTraining completed!")
    print(f"Total datasets trained: {len(TRAIN_DATASETS)}")
    print(f"Total images processed: {sum(dataset_sizes)}")
    print(f"Total training time: {sum(training_times):.2f} seconds")


if __name__ == "__main__":
    main()