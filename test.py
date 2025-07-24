"""
Testing script for image authenticity detection model
"""
import joblib
import numpy as np
import time
import warnings
from config import *
from utils import get_image_paths, calculate_metrics
from model import batch_detect_images

# Ignore specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")


def print_and_save_results(file, real_dataset, fake_dataset, time_taken,
                           real_acc, fake_acc, metrics):
    """Print and save results for a dataset pair"""
    print(f"Real image accuracy: {real_acc:.4f}")
    print(f"Fake image accuracy: {fake_acc:.4f}")
    print(f"Overall metrics:")
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")

    # Write to file
    file.write(f"Real dataset: {real_dataset}\n")
    file.write(f"Fake dataset: {fake_dataset}\n")
    file.write(f"Processing time: {time_taken:.2f} seconds\n")
    file.write(f"Real image accuracy: {real_acc:.4f}\n")
    file.write(f"Fake image accuracy: {fake_acc:.4f}\n")
    file.write(f"Overall metrics:\n")
    for metric, value in metrics.items():
        file.write(f"  {metric.upper()}: {value:.4f}\n")
    file.write("\n")


def save_summary(model_filename, start_time, all_y_true, all_y_pred,
                 all_y_proba, all_metrics):
    """Save summary results"""
    total_time = time.time() - start_time

    if len(all_metrics) > 0:
        # Calculate average metrics
        avg_metrics = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'ap']:
            avg_metrics[metric] = np.mean([m[metric] for m in all_metrics])

        # Calculate overall metrics
        overall_metrics = calculate_metrics(all_y_true, all_y_pred, all_y_proba)

        # Print summary
        print(f"\n{'=' * 20} Model {model_filename} Summary {'=' * 20}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Total images processed: {len(all_y_true)}")
        print(f"Average processing speed: {len(all_y_true) / total_time:.2f} images/second")

        # Save to file
        summary_filename = f'summary_metrics_{model_filename[:-4]}.txt'
        with open(summary_filename, 'w', encoding='utf-8') as file:
            file.write(f"Model: {model_filename}\n")
            file.write(f"Total processing time: {total_time:.2f} seconds\n")
            file.write(f"Total images processed: {len(all_y_true)}\n")
            file.write(f"Average processing speed: {len(all_y_true) / total_time:.2f} images/second\n")
            file.write(f"Number of dataset pairs processed: {len(all_metrics)}\n")
            file.write(f"Number of threads used: {MAX_WORKERS}\n")
            file.write(f"Batch size: {BATCH_SIZE}\n\n")

            file.write("Average metrics (across dataset pairs):\n")
            for metric, value in avg_metrics.items():
                file.write(f"  {metric.upper()}: {value:.4f}\n")

            file.write("\nOverall metrics (all data):\n")
            for metric, value in overall_metrics.items():
                file.write(f"  {metric.upper()}: {value:.4f}\n")


def main():
    """Main testing function"""
    print(f"Performance configuration: Using {MAX_WORKERS} threads, batch size: {BATCH_SIZE}")

    for model_filename in MODEL_FILENAMES:
        print(f"\nUsing model: {model_filename}")
        total_start_time = time.time()

        # Load model
        print(f"Loading model from {model_filename}")
        try:
            model, scaler, selector = joblib.load(model_filename)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {e}")
            continue

        # Initialize result collections
        all_y_true = []
        all_y_pred = []
        all_y_proba = []
        all_metrics = []

        result_filename = f'detection_results_{model_filename[:-4]}.txt'
        with open(result_filename, 'w', encoding='utf-8') as result_file:
            # Process each dataset pair
            for i, (real_dataset, fake_dataset) in enumerate(zip(REAL_DATASETS, FAKE_DATASETS)):
                print(f"\nProcessing dataset pair {i + 1}/{len(REAL_DATASETS)}:")
                print(f"Real dataset: {real_dataset}")
                print(f"Fake dataset: {fake_dataset}")

                pair_start_time = time.time()

                # Lists for this dataset pair
                pair_y_true = []
                pair_y_pred = []
                pair_y_proba = []

                # Get image paths
                real_paths = get_image_paths(real_dataset)
                fake_paths = get_image_paths(fake_dataset)

                print(f"Found {len(real_paths)} real images, {len(fake_paths)} fake images")

                # Process real images
                if real_paths:
                    print("Processing real images...")
                    _, predictions, probabilities = batch_detect_images(
                        real_paths, model, scaler, selector, KERNEL_SIZES,
                        max_workers=MAX_WORKERS, batch_size=BATCH_SIZE
                    )

                    pair_y_true.extend([1] * len(predictions))  # Label 1 for real
                    pair_y_pred.extend(predictions)
                    pair_y_proba.extend(probabilities)

                # Process fake images
                if fake_paths:
                    print("Processing fake images...")
                    _, predictions, probabilities = batch_detect_images(
                        fake_paths, model, scaler, selector, KERNEL_SIZES,
                        max_workers=MAX_WORKERS, batch_size=BATCH_SIZE
                    )

                    pair_y_true.extend([-1] * len(predictions))  # Label -1 for fake
                    pair_y_pred.extend(predictions)
                    pair_y_proba.extend(probabilities)

                pair_time = time.time() - pair_start_time
                print(f"Dataset pair processing time: {pair_time:.2f} seconds")

                # Calculate and save metrics
                if len(pair_y_true) > 0:
                    pair_metrics = calculate_metrics(pair_y_true, pair_y_pred, pair_y_proba)
                    all_metrics.append(pair_metrics)

                    # Add to overall collections
                    all_y_true.extend(pair_y_true)
                    all_y_pred.extend(pair_y_pred)
                    all_y_proba.extend(pair_y_proba)

                    # Calculate per-class accuracies
                    real_correct = sum(1 for true, pred in zip(pair_y_true, pair_y_pred)
                                       if true == 1 and pred == 1)
                    fake_correct = sum(1 for true, pred in zip(pair_y_true, pair_y_pred)
                                       if true == -1 and pred == -1)
                    real_total = sum(1 for true in pair_y_true if true == 1)
                    fake_total = sum(1 for true in pair_y_true if true == -1)

                    real_acc = real_correct / real_total if real_total > 0 else 0
                    fake_acc = fake_correct / fake_total if fake_total > 0 else 0

                    # Print and save results
                    print_and_save_results(
                        result_file, real_dataset, fake_dataset, pair_time,
                        real_acc, fake_acc, pair_metrics
                    )
                else:
                    print("Warning: No images were successfully processed for this dataset pair")

        # Save summary
        save_summary(model_filename, total_start_time, all_y_true, all_y_pred,
                     all_y_proba, all_metrics)

    print("\nTesting completed!")


if __name__ == "__main__":
    main()