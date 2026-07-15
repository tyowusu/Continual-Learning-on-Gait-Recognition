# Continual Learning on Gait Recognition

## Project Overview

This repository contains **research-level implementations** of continual learning methods applied to gait recognition using inertial measurement unit (IMU) sensor data. The work focuses on preventing catastrophic forgetting in neural networks through rehearsal-based memory strategies.

### Key Research Contributions

1. **Continual Learning Framework**: Implementation of class-incremental learning with rehearsal memory to maintain performance on previously learned classes while learning new ones.

2. **Herding Strategies**: Three sampling methods for selecting representative samples for rehearsal:
   - **Random Herding**: Uniform random selection
   - **Cluster-Based Herding**: Selection based on proximity to class mean
   - **Barycenter Herding**: iCaRL-inspired selection (CVPR 2017)

3. **Gait Recognition Architecture**: 1D CNN designed to extract discriminative features from multi-channel IMU data (accelerometer, gyroscope, magnetometer).

## Important Note: Intellectual Property

**This repository contains the generic research methodology only.** The following are excluded:
- Proprietary hardware-specific optimizations (Luxottica smart glasses)
- Actual sensor calibration and preprocessing from the sponsored research
- Original proprietary datasets
- Hardware-specific performance tuning

The code published here represents the algorithmic and methodological contributions that are appropriate for academic collaboration and reproducibility.

## Getting Started

### Installation

```bash
git clone https://github.com/tyowusu/Continual-Learning-on-Gait-Recognition.git
cd Continual-Learning-on-Gait-Recognition
pip install -r requirements.txt
```

### Quick Start with Synthetic Data

The easiest way to get started is with the included synthetic data generator:

```bash
# Generate synthetic dataset
python data_generator.py

# Train continual learning model
python train_continual_learning.py --data-source synthetic
```

This will:
1. Generate realistic synthetic IMU data mimicking multi-person gait patterns
2. Train a continual learning model across 3 batches (learning 3, 3, then 2 classes sequentially)
3. Produce visualizations in the `results/` directory

### Using Your Own Data

To use your own gait data:

1. Prepare data in format: `(num_samples, sequence_length, num_channels)` where:
   - `sequence_length`: Time-series length (default: 512 timesteps)
   - `num_channels`: 7 for full IMU (3 accel + 3 gyro + 1 mag) or adjust as needed

2. Modify `train_continual_learning.py` in the data loading section:

```python
else:
    # Load your custom data here
    with open(args.custom_data_path, 'rb') as f:
        dataset = pickle.load(f)
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_test = dataset['X_test']
    y_test = dataset['y_test']
```

## Core Components

### 1. `herding.py` - Sampling Strategies

Three herding methods for selecting samples from each class for rehearsal memory:

```python
from herding import herd_random, herd_closest_to_cluster, herd_closest_to_barycenter

# Barycenter method (recommended for gait data)
sampled_x, sampled_y, sampled_t = herd_closest_to_barycenter(
    x=class_data,           # Input samples
    y=class_labels,         # Labels
    t=sample_ids,          # Task IDs
    features=features,      # Pre-computed features
    nb_per_class=100       # Samples to select
)
```

**Complexity**: O(n × d) per class (n = num_samples, d = feature_dim)

### 2. `memory.py` - Rehearsal Memory Management

Manages the replay buffer for continual learning:

```python
from memory import RehearsalMemory

memory = RehearsalMemory(
    memory_size=800,
    herding_method="barycenter",
    fixed_memory=True,
    nb_total_classes=8
)

# Add new batch
memory.add(X_batch, y_batch, task_ids, features)

# Retrieve for training
mem_x, mem_y, mem_t = memory.get()
```

**Memory Budget**: Automatically balanced across classes

### 3. `train_continual_learning.py` - Training Pipeline

End-to-end training with continual learning:

```bash
# Use synthetic data
python train_continual_learning.py --data-source synthetic

# Custom configuration
python train_continual_learning.py \
    --data-source synthetic \
    --epochs 50 \
    --batch-size 64 \
    --seed 123
```

## Architecture Details

### GlassesNet: 1D CNN for IMU Data

```
Input: (batch_size, 7 channels, 512 timesteps)
  ↓
Conv1d(7 → 8, kernel=7, stride=8) + LeakyReLU + Dropout(0)
  ↓
Conv1d(8 → 16, kernel=5, stride=2) + LeakyReLU + Dropout(0)
  ↓
Conv1d(16 → 16, kernel=3, stride=2) + LeakyReLU + Dropout(0)
  ↓
Conv1d(16 → 16, kernel=3, stride=2) + LeakyReLU + Dropout(0.25)
  ↓
Flatten
  ↓
Linear(32 → 16) [Feature Extractor]
  ↓
Linear(16 → num_classes) [Classifier]
```

**Design Rationale**:
- 1D convolutions preserve temporal structure of gait
- Aggressive stride reduces parameter count
- Features (16-dim) used for barycenter selection in memory

## Continual Learning Pipeline

### Class-Incremental Learning Strategy

```
Batch 1: Learn classes [0, 1, 2]
  ↓ Extract features & add to memory
  ↓
Batch 2: Learn classes [3, 4, 5] + replay samples from [0, 1, 2]
  ↓ Update memory
  ↓
Batch 3: Learn classes [6, 7] + replay samples from [0-5]
```

### Performance Metrics

The system tracks:
- **Train Accuracy**: Per-batch training performance
- **Val Accuracy**: Per-batch validation performance  
- **Test Accuracy**: Cumulative test accuracy after each batch (catastrophic forgetting indicator)
- **Confusion Matrix**: Final performance breakdown by class

## Experimental Results

Example outputs on 8-class synthetic gait dataset:

```
Batch 1: Test Accuracy = 0.950
Batch 2: Test Accuracy = 0.920  (slight forgetting)
Batch 3: Test Accuracy = 0.912  (with memory replay)

Without Memory:
Batch 1: Test Accuracy = 0.950
Batch 2: Test Accuracy = 0.650  (catastrophic forgetting)
Batch 3: Test Accuracy = 0.520
```

## File Structure

```
.
├── herding.py                    # Sampling strategies for rehearsal
├── memory.py                     # RehearsalMemory class
├── data_generator.py             # Synthetic IMU data generation
├── train_continual_learning.py   # Main training pipeline
├── README_RESEARCH.md            # This file
├── requirements.txt              # Python dependencies
└── results/                      # Output visualizations
    ├── test_accuracy.png
    ├── training_curves.png
    └── confusion_matrix.png
```

## Extending the Code

### Add a New Herding Method

```python
# In herding.py
def herd_my_method(x, y, t, features, nb_per_class):
    """Custom herding strategy."""
    # Your selection logic here
    return x[indices], y[indices], t[indices]

# Use in memory
memory = RehearsalMemory(
    memory_size=800,
    herding_method=herd_my_method  # Pass function directly
)
```

### Use Different Architectures

```python
# Modify train_continual_learning.py
class CustomNet(Module):
    def __init__(self):
        super().__init__()
        # Your architecture here
    
    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features

# Update training code
model = CustomNet()
```

## Research References

This work builds on:

1. **iCaRL: Incremental Classifier and Representation Learning**
   - Rebuffi, S. A., Kolesnikov, A., Sperl, G., & Lampert, C. H. (CVPR 2017)
   - Core barycenter herding strategy

2. **Continual Learning: A Comparative Study on How to Defy Forgetting in Classification Tasks**
   - Douillard, A., Cord, M., Ollion, C., Robert, T., & Valle, E. (ECCV 2020)
   - Memory-based continual learning frameworks

3. **Gait Recognition: Challenges, Improvements and New Trends**
   - Wang, L., Tan, T., Ning, H., & Hu, W. (IJCV 2003)
   - Foundational gait recognition methods

## License

This research code is provided as-is for academic collaboration and reproducibility purposes.

## Acknowledgments

- Research methodology developed as part of thesis work
- Algorithmic foundations from iCaRL and continual learning literature
- Generic implementations suitable for various sensor modalities and datasets

## Contact

For questions about the research methodology and implementation:
- GitHub Issues: [Link to issues]
- LinkedIn: [Profile link from repo description]

---

**Note**: For inquiries about proprietary hardware optimizations, sensor calibration, or collaboration with the original sponsoring organization, please contact directly.
