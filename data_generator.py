"""
Synthetic Gait Data Generator

This module generates synthetic sensor data that mimics the structure and characteristics
of inertial measurement unit (IMU) data from wearable devices (e.g., smart glasses).

The synthetic data is designed to:
- Provide reproducible datasets for algorithm development and testing
- Enable running the training pipeline without proprietary hardware data
- Facilitate research collaboration on continual learning methods

Note: For production use, replace with actual sensor data collection.
"""

import numpy as np
from typing import Dict, Tuple, List
import pickle


def generate_synthetic_gait_sequence(
    num_samples: int = 100,
    sequence_length: int = 512,
    num_channels: int = 7,
    gait_pattern_variance: float = 0.1,
    class_id: int = 0,
    random_seed: int = None
) -> np.ndarray:
    """
    Generate synthetic IMU sensor data simulating gait patterns.
    
    The 7 channels represent:
      - Channels 0-2: Accelerometer (x, y, z axes)
      - Channels 3-5: Gyroscope (x, y, z axes)
      - Channel 6: Magnetometer magnitude
    
    Args:
        num_samples: Number of individual gait cycles to generate
        sequence_length: Length of each sensor sequence (time steps)
        num_channels: Number of sensor channels (7: accel + gyro + mag)
        gait_pattern_variance: Controls variation across samples (0.0-1.0)
        class_id: Class identifier (used to add class-specific bias)
        random_seed: Seed for reproducibility
    
    Returns:
        np.ndarray: Synthetic data of shape (num_samples, sequence_length, num_channels)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    data = np.zeros((num_samples, sequence_length, num_channels))
    
    for i in range(num_samples):
        # Create base gait pattern (periodic signal)
        t = np.linspace(0, 4 * np.pi, sequence_length)
        
        # Accelerometer channels (sinusoidal gait pattern with class-specific offset)
        accel_base_freq = 2.0 + (class_id % 8) * 0.1  # Slight variation per class
        data[i, :, 0] = 9.81 + 5 * np.sin(accel_base_freq * t) + np.random.normal(0, 0.5, sequence_length)
        data[i, :, 1] = 5 * np.cos(accel_base_freq * t) + np.random.normal(0, 0.3, sequence_length)
        data[i, :, 2] = 9.81 + 3 * np.sin(accel_base_freq * t + np.pi/4) + np.random.normal(0, 0.4, sequence_length)
        
        # Gyroscope channels (rotational motion during gait)
        data[i, :, 3] = 50 * np.sin(accel_base_freq * t) + np.random.normal(0, 5, sequence_length)
        data[i, :, 4] = 50 * np.cos(accel_base_freq * t) + np.random.normal(0, 5, sequence_length)
        data[i, :, 5] = 30 * np.sin(accel_base_freq * t + np.pi/2) + np.random.normal(0, 3, sequence_length)
        
        # Magnetometer magnitude
        data[i, :, 6] = 50 + 10 * np.sin(t * 0.5) + np.random.normal(0, 2, sequence_length)
        
        # Add class-specific variations (e.g., gait signature)
        class_variation = (class_id % 8) / 8.0
        data[i] *= (1.0 + class_variation * 0.1)
        
        # Add inter-sample variation
        variation = 1.0 + np.random.normal(0, gait_pattern_variance, (1, num_channels))
        data[i] *= variation
    
    return data


def generate_synthetic_gait_dataset(
    num_classes: int = 8,
    samples_per_class: int = 100,
    sequence_length: int = 512,
    num_channels: int = 7,
    train_ratio: float = 0.9,
    random_seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Generate a complete synthetic gait dataset with train/test splits.
    
    Args:
        num_classes: Number of gait classes (individuals)
        samples_per_class: Samples per class in training set
        sequence_length: Length of each sensor sequence
        num_channels: Number of sensor channels (7)
        train_ratio: Proportion of data for training (vs testing)
        random_seed: Seed for reproducibility
    
    Returns:
        Dictionary containing:
          - 'X_train': Training data (num_train_samples, sequence_length, num_channels)
          - 'y_train': Training labels (num_train_samples,)
          - 'X_test': Test data (num_test_samples, sequence_length, num_channels)
          - 'y_test': Test labels (num_test_samples,)
          - 'class_names': List of class labels
    """
    np.random.seed(random_seed)
    
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    
    for class_id in range(num_classes):
        # Generate all samples for this class
        all_samples = generate_synthetic_gait_sequence(
            num_samples=samples_per_class * 2,  # Generate extra for train/test split
            sequence_length=sequence_length,
            num_channels=num_channels,
            class_id=class_id,
            random_seed=random_seed + class_id
        )
        
        # Split into train and test
        split_point = int(len(all_samples) * train_ratio)
        X_train_list.append(all_samples[:split_point])
        y_train_list.extend([class_id] * split_point)
        
        X_test_list.append(all_samples[split_point:])
        y_test_list.extend([class_id] * (len(all_samples) - split_point))
    
    # Concatenate all classes
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.array(y_train_list)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.array(y_test_list)
    
    # Shuffle
    train_indices = np.random.permutation(len(X_train))
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    
    test_indices = np.random.permutation(len(X_test))
    X_test = X_test[test_indices]
    y_test = y_test[test_indices]
    
    class_names = [f"Individual_{i}" for i in range(num_classes)]
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'class_names': class_names
    }


def save_synthetic_dataset(dataset: Dict, filepath: str = "synthetic_gait_data.pickle"):
    """Save synthetic dataset to pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Synthetic dataset saved to {filepath}")


def load_synthetic_dataset(filepath: str = "synthetic_gait_data.pickle") -> Dict:
    """Load synthetic dataset from pickle file."""
    with open(filepath, "rb") as f:
        dataset = pickle.load(f)
    print(f"Synthetic dataset loaded from {filepath}")
    return dataset


if __name__ == "__main__":
    # Example: Generate and save synthetic dataset
    print("Generating synthetic gait dataset...")
    dataset = generate_synthetic_gait_dataset(
        num_classes=8,
        samples_per_class=150,
        sequence_length=512,
        num_channels=7,
        train_ratio=0.9,
        random_seed=42
    )
    
    print(f"Training data shape: {dataset['X_train'].shape}")
    print(f"Test data shape: {dataset['X_test'].shape}")
    print(f"Classes: {dataset['class_names']}")
    
    # Save for later use
    save_synthetic_dataset(dataset, "synthetic_gait_data.pickle")
