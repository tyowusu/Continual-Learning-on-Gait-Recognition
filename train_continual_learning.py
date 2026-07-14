"""
Continual Learning Training Pipeline for Gait Recognition

This script implements a continual learning approach to gait recognition using
rehearsal-based memory to prevent catastrophic forgetting.

Research Methodology:
- Model: 1D CNN feature extractor on IMU sensor data
- Continual Learning: Class-incremental learning with rehearsal memory
- Herding Methods: Barycenter selection (based on iCaRL)

This implementation is designed to work with:
1. Synthetic data (included for reproducibility)
2. Your own gait sensor data (provide your data loading function)
3. Public gait datasets (CASIA-B, OU-ISIR, etc.)

NOTE: This is the generic research methodology. Proprietary hardware-specific
optimizations and actual Luxottica data are excluded to respect IP constraints.
"""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.nn import Module, LeakyReLU, Dropout, Conv1d, Flatten, Linear
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from torch.optim import Adam
import seaborn as sns
from typing import Tuple, List, Optional, Dict
import argparse

# Import the generic rehearsal memory and herding methods
from memory import RehearsalMemory
import herding

# Import synthetic data generator
from data_generator import generate_synthetic_gait_dataset, save_synthetic_dataset


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed = 42
    epochs_per_batch = 30
    batch_size = 32
    learning_rate = 0.001
    memory_size = 800
    herding_method = "barycenter"
    fixed_memory = True
    nb_total_classes = 8
    
    # For synthetic data
    num_classes = 8
    samples_per_class = 150
    sequence_length = 512
    num_channels = 7  # Accel (3) + Gyro (3) + Mag (1)
    
    # Continual learning batches: [3, 3, 2] means
    # Batch 1: Learn 3 classes, Batch 2: Learn 3 new classes, Batch 3: Learn 2 new classes
    batch_order = [3, 3, 2]


# ============================================================================
# DATA HANDLING
# ============================================================================

class SimpleDataset(Dataset):
    """PyTorch dataset wrapper for gait sensor data."""
    
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        Args:
            x: Sensor data of shape (num_samples, sequence_length, num_channels)
            y: Labels of shape (num_samples,)
        """
        super(SimpleDataset, self).__init__()
        # Convert to (num_samples, num_channels, sequence_length) for Conv1d
        self.x = torch.tensor(x).permute(0, 2, 1).float()
        self.y = torch.tensor(y).long()
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def normalize_input(x: np.ndarray) -> np.ndarray:
    """
    Normalize sensor data by removing per-sample mean.
    
    Args:
        x: Input data of shape (num_samples, sequence_length, num_channels)
    
    Returns:
        Normalized data
    """
    m = np.mean(x, axis=1, keepdims=True)
    return x - m


def prepare_continual_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_order: List[int],
    random_seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Prepare class-incremental batches for continual learning.
    
    Args:
        X: Input data of shape (num_samples, sequence_length, num_channels)
        y: Labels of shape (num_samples,)
        batch_order: List of class counts per batch, e.g., [3, 3, 2]
        random_seed: For reproducibility
    
    Returns:
        List of (X_batch, y_batch) tuples
    """
    np.random.seed(random_seed)
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Group by class
    batches = []
    class_indices = {}
    for class_id in np.unique(y):
        class_indices[class_id] = np.where(y == class_id)[0]
    
    # Create batches according to batch_order
    batch_classes = []
    current_class_id = 0
    for num_classes_in_batch in batch_order:
        batch_class_ids = list(range(current_class_id, current_class_id + num_classes_in_batch))
        batch_classes.append(batch_class_ids)
        current_class_id += num_classes_in_batch
    
    # Extract data for each batch
    for batch_class_ids in batch_classes:
        batch_indices = []
        for class_id in batch_class_ids:
            batch_indices.extend(class_indices[class_id])
        
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        batches.append((X_batch, y_batch))
    
    return batches


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class GlassesNet(Module):
    """
    1D CNN for gait recognition from IMU sensor data.
    
    Architecture:
    - Input: (batch_size, 7 channels, 512 timesteps)
    - Conv1d layers with LeakyReLU activations
    - Dropout for regularization
    - FC layer for feature extraction (output: 16-dim features)
    
    This architecture is designed to extract discriminative features
    from accelerometer, gyroscope, and magnetometer data.
    """
    
    def __init__(self, input_channels: int = 7, num_classes: int = 8):
        """
        Args:
            input_channels: Number of input channels (7 for accel+gyro+mag)
            num_classes: Number of output classes (for classification head)
        """
        super(GlassesNet, self).__init__()
        self.lrelu = LeakyReLU()
        self.dropout1 = Dropout(0.0)
        self.dropout2 = Dropout(0.0)
        self.dropout3 = Dropout(0.0)
        self.dropout4 = Dropout(0.25)
        
        # 1D Convolutional layers
        self.layer1 = Conv1d(input_channels, 8, kernel_size=7, stride=8)
        self.layer2 = Conv1d(8, 16, kernel_size=5, stride=2)
        self.layer3 = Conv1d(16, 16, kernel_size=3, stride=2)
        self.layer4 = Conv1d(16, 16, kernel_size=3, stride=2)
        
        self.flatten = Flatten()
        self.fc = Linear(32, 16)  # Feature extractor
        self.classifier = Linear(16, num_classes)  # Classification head
    
    def forward(self, x):
        """Forward pass."""
        y = self.dropout1(self.lrelu(self.layer1(x)))
        y = self.dropout2(self.lrelu(self.layer2(y)))
        y = self.dropout3(self.lrelu(self.layer3(y)))
        y = self.dropout4(self.lrelu(self.layer4(y)))
        y = self.flatten(y)
        features = self.fc(y)  # 16-dim features
        logits = self.classifier(features)  # Classification
        return logits, features


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(
    model: Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Module,
    device: torch.device
) -> float:
    """
    Train for one epoch.
    
    Returns:
        Average accuracy for the epoch
    """
    model.train()
    total_correct = 0
    total_samples = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        logits, _ = model(X_batch)
        loss = loss_fn(logits, y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        predictions = torch.argmax(logits, dim=1)
        total_correct += torch.sum(predictions == y_batch).item()
        total_samples += len(y_batch)
    
    return total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate model on test set.
    
    Returns:
        (accuracy, loss)
    """
    model.eval()
    X_test_tensor = torch.tensor(X_test).permute(0, 2, 1).to(device).float()
    y_test_tensor = torch.tensor(y_test).to(device).long()
    
    logits, _ = model(X_test_tensor)
    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, y_test_tensor).item()
    
    accuracy = torch.mean(
        (torch.argmax(logits, dim=1) == y_test_tensor).float()
    ).item()
    
    return accuracy, loss


def extract_features(
    model: Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 32
) -> np.ndarray:
    """
    Extract feature representations from model.
    
    Returns:
        Features of shape (num_samples, 16)
    """
    model.eval()
    features_list = []
    
    X_tensor = torch.tensor(X).permute(0, 2, 1).to(device).float()
    dataset = torch.utils.data.TensorDataset(X_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    with torch.no_grad():
        for (X_batch,) in dataloader:
            _, feats = model(X_batch)
            features_list.append(feats.cpu().numpy())
    
    return np.concatenate(features_list, axis=0)


# ============================================================================
# CONTINUAL LEARNING TRAINING
# ============================================================================

def train_continual_learning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Config,
    use_memory: bool = True
) -> Dict:
    """
    Train model using continual learning with rehearsal memory.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        config: Training configuration
        use_memory: Whether to use rehearsal memory
    
    Returns:
        Dictionary with results and metrics
    """
    # Set seeds
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # Prepare continual batches
    print("Preparing continual learning batches...")
    continual_batches = prepare_continual_batches(
        X_train, y_train, config.batch_order, random_seed=config.seed
    )
    
    # Initialize model
    model = GlassesNet(
        input_channels=config.num_channels,
        num_classes=config.nb_total_classes
    )
    model.to(config.device)
    
    # Initialize rehearsal memory
    if use_memory:
        memory = RehearsalMemory(
            memory_size=config.memory_size,
            herding_method=config.herding_method,
            fixed_memory=config.fixed_memory,
            nb_total_classes=config.nb_total_classes
        )
    
    # Training history
    history = {
        'batch_accuracies': [],
        'batch_train_curves': [],
        'batch_val_curves': [],
        'test_accuracies': [],
        'confusion_matrices': []
    }
    
    # Train on each batch
    for batch_idx, (X_batch, y_batch) in enumerate(continual_batches):
        print(f"\n{'='*60}")
        print(f"BATCH {batch_idx + 1}/{len(continual_batches)}")
        print(f"{'='*60}")
        print(f"Learning {len(np.unique(y_batch))} new classes: {np.unique(y_batch)}")
        
        # Normalize
        X_batch = normalize_input(X_batch)
        
        # Split batch into train/val
        X_batch_train, X_batch_val, y_batch_train, y_batch_val = train_test_split(
            X_batch, y_batch, test_size=0.1, shuffle=True,
            stratify=y_batch, random_state=config.seed
        )
        
        # Add rehearsal memory if available
        if use_memory and batch_idx > 0:
            print(f"Adding {len(memory)} rehearsal samples to training...")
            mem_x, mem_y, mem_t = memory.get()
            X_batch_train = np.concatenate([X_batch_train, mem_x], axis=0)
            y_batch_train = np.concatenate([y_batch_train, mem_y], axis=0)
        
        # Create dataloader
        dataset = SimpleDataset(X_batch_train, y_batch_train)
        dataloader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
        )
        
        # Training
        optimizer = Adam(model.parameters(), lr=config.learning_rate)
        loss_fn = CrossEntropyLoss()
        
        train_curve = []
        val_curve = []
        
        for epoch in range(config.epochs_per_batch):
            train_acc = train_epoch(model, dataloader, optimizer, loss_fn, config.device)
            val_acc, val_loss = evaluate(
                model, normalize_input(X_batch_val), y_batch_val, config.device
            )
            
            train_curve.append(train_acc)
            val_curve.append(val_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:2d}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
        
        # Store curves
        history['batch_train_curves'].append(train_curve)
        history['batch_val_curves'].append(val_curve)
        
        # Evaluate on full test set
        test_acc, test_loss = evaluate(
            model, normalize_input(X_test), y_test, config.device
        )
        history['test_accuracies'].append(test_acc)
        print(f"\nTest Accuracy after Batch {batch_idx + 1}: {test_acc:.3f}")
        
        # Update memory with new data (for next batch)
        if use_memory:
            print("Updating rehearsal memory...")
            features = extract_features(model, normalize_input(X_batch), config.device)
            memory.add(
                X_batch.astype(np.float32),
                y_batch.astype(np.int64),
                np.arange(len(X_batch)).astype(np.int64),
                features.astype(np.float32)
            )
        
        # Generate confusion matrix
        model.eval()
        X_test_normalized = normalize_input(X_test)
        X_test_tensor = torch.tensor(X_test_normalized).permute(0, 2, 1).to(config.device).float()
        with torch.no_grad():
            logits, _ = model(X_test_tensor)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
        
        cm = confusion_matrix(y_test, predictions)
        history['confusion_matrices'].append(cm)
        history['batch_accuracies'].append(test_acc)
    
    return history, model


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(history: Dict, save_path: str = "results"):
    """Visualize training results."""
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # Plot test accuracy across batches
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history['test_accuracies']) + 1),
             history['test_accuracies'], marker='o', linewidth=2, markersize=8)
    plt.xlabel("Batch Number", fontsize=12)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.title("Test Accuracy Across Continual Learning Batches", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/test_accuracy.png", dpi=150)
    plt.close()
    
    # Plot training curves per batch
    fig, axes = plt.subplots(1, len(history['batch_train_curves']), figsize=(15, 4))
    for i, (train_curve, val_curve) in enumerate(
        zip(history['batch_train_curves'], history['batch_val_curves'])
    ):
        if len(history['batch_train_curves']) > 1:
            ax = axes[i]
        else:
            ax = axes
        ax.plot(train_curve, label='Train', linewidth=2)
        ax.plot(val_curve, label='Val', linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Batch {i+1}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}/training_curves.png", dpi=150)
    plt.close()
    
    # Plot final confusion matrix
    final_cm = history['confusion_matrices'][-1]
    plt.figure(figsize=(10, 8))
    sns.heatmap(final_cm, annot=True, fmt='.2f', cmap='Blues', cbar=True)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Final Confusion Matrix (after all batches)")
    plt.tight_layout()
    plt.savefig(f"{save_path}/confusion_matrix.png", dpi=150)
    plt.close()
    
    print(f"Results saved to {save_path}/")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Continual Learning for Gait Recognition"
    )
    parser.add_argument(
        "--data-source", type=str, default="synthetic",
        choices=["synthetic", "custom"],
        help="Data source: synthetic or custom path"
    )
    parser.add_argument(
        "--custom-data-path", type=str, default=None,
        help="Path to custom gait data pickle file"
    )
    parser.add_argument(
        "--use-memory", action="store_true", default=True,
        help="Use rehearsal memory for continual learning"
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Epochs per batch"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    config = Config()
    config.seed = args.seed
    config.epochs_per_batch = args.epochs
    config.batch_size = args.batch_size
    
    print("="*60)
    print("CONTINUAL LEARNING FOR GAIT RECOGNITION")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Seed: {config.seed}")
    
    # Load or generate data
    if args.data_source == "synthetic":
        print("\nGenerating synthetic gait dataset...")
        dataset = generate_synthetic_gait_dataset(
            num_classes=config.num_classes,
            samples_per_class=config.samples_per_class,
            sequence_length=config.sequence_length,
            num_channels=config.num_channels,
            random_seed=config.seed
        )
        X_train = dataset['X_train']
        y_train = dataset['y_train']
        X_test = dataset['X_test']
        y_test = dataset['y_test']
        
        # Save for reproducibility
        save_synthetic_dataset(dataset, "synthetic_gait_data.pickle")
        print(f"Synthetic data: {X_train.shape}, saved to synthetic_gait_data.pickle")
    
    else:
        raise NotImplementedError(
            "Custom data loading not implemented. "
            "Provide your own data loading logic here."
        )
    
    # Train continual learning model
    print("\nStarting continual learning training...")
    history, model = train_continual_learning(
        X_train, y_train, X_test, y_test, config, use_memory=args.use_memory
    )
    
    # Visualize results
    print("\nVisualizing results...")
    visualize_results(history, save_path="results")
    
    print("\n" + "="*60)
    print(f"Final Test Accuracy: {history['test_accuracies'][-1]:.3f}")
    print(f"Batch Accuracies: {[f'{acc:.3f}' for acc in history['batch_accuracies']]}")
    print("="*60)


if __name__ == "__main__":
    main()
