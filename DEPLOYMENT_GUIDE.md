# Deployment & Integration Guide

This guide explains how to integrate the continual learning pipeline into your own projects or production systems.

## Setup for Custom Environments

### 1. Minimal Dependencies

If you only need the core algorithms (without visualization):

```bash
pip install torch numpy scikit-learn
```

### 2. Development Setup

```bash
git clone https://github.com/tyowusu/Continual-Learning-on-Gait-Recognition.git
cd Continual-Learning-on-Gait-Recognition
pip install -r requirements.txt
```

### 3. GPU Acceleration (Optional)

For CUDA-enabled training:

```bash
# CUDA 11.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or use your specific CUDA version
# Visit: https://pytorch.org/get-started/locally/
```

## Integration Patterns

### Pattern 1: Standalone Training Script

```python
from train_continual_learning import (
    train_continual_learning,
    Config,
    visualize_results
)
from data_generator import generate_synthetic_gait_dataset

# Configure
config = Config()
config.epochs_per_batch = 50
config.batch_size = 64

# Generate or load data
dataset = generate_synthetic_gait_dataset(num_classes=8)
X_train, y_train = dataset['X_train'], dataset['y_train']
X_test, y_test = dataset['X_test'], dataset['y_test']

# Train
history, model = train_continual_learning(
    X_train, y_train, X_test, y_test, config
)

# Visualize
visualize_results(history)
```

### Pattern 2: Custom Data Loader Integration

```python
import numpy as np
from train_continual_learning import train_continual_learning, Config

class CustomGaitDataset:
    def __init__(self, data_path):
        # Load your proprietary data
        self.data = self._load_data(data_path)
    
    def _load_data(self, path):
        # Your loading logic
        pass
    
    def get_continual_batches(self, batch_order=[3, 3, 2]):
        # Return list of (X_batch, y_batch) tuples
        pass

# Use it
dataset = CustomGaitDataset("/path/to/data")
X_train, y_train = dataset.get_training_data()

config = Config()
history, model = train_continual_learning(X_train, y_train, X_test, y_test, config)
```

### Pattern 3: Feature Extraction Only

```python
from train_continual_learning import GlassesNet, extract_features
import torch

# Load pre-trained model (or train from scratch)
model = GlassesNet(input_channels=7, num_classes=8)
model.load_state_dict(torch.load('model_checkpoint.pt'))

# Extract features for downstream tasks
features = extract_features(model, X_data, device=torch.device('cuda'))
print(f"Features shape: {features.shape}")  # (num_samples, 16)

# Use features for:
# - Clustering
# - Anomaly detection
# - Similarity search
# - Classification with different ML models
```

### Pattern 4: Memory Management Only

```python
from memory import RehearsalMemory
from herding import herd_closest_to_barycenter

# Create memory manager
memory = RehearsalMemory(
    memory_size=1000,
    herding_method="barycenter",
    fixed_memory=True,
    nb_total_classes=10
)

# Add samples
memory.add(X_new, y_new, task_ids, features)

# Retrieve for training
mem_x, mem_y, mem_t = memory.get()

# Slice for specific classes
class_subset = memory.slice(keep_classes=[0, 1, 2])

# Save/load persistent memory
memory.save('checkpoint.npz')
memory.load('checkpoint.npz')
```

## Serving Models

### Option 1: PyTorch Model Server

```python
import torch
from train_continual_learning import GlassesNet

# Load trained model
model = GlassesNet(input_channels=7, num_classes=8)
model.load_state_dict(torch.load('final_model.pt'))
model.eval()

# Export for inference
trace = torch.jit.trace(
    model,
    torch.randn(1, 7, 512)  # Dummy input
)
trace.save('gait_model.pt')
```

### Option 2: ONNX Export

```python
import torch
import torch.onnx
from train_continual_learning import GlassesNet

model = GlassesNet(input_channels=7, num_classes=8)
model.load_state_dict(torch.load('final_model.pt'))
model.eval()

dummy_input = torch.randn(1, 7, 512)
torch.onnx.export(
    model, dummy_input, "gait_model.onnx",
    input_names=['imu_data'],
    output_names=['logits', 'features'],
    dynamic_axes={'imu_data': {0: 'batch_size'}}
)
```

### Option 3: REST API with Flask

```python
from flask import Flask, request, jsonify
import torch
import numpy as np
from train_continual_learning import GlassesNet

app = Flask(__name__)
model = GlassesNet(input_channels=7, num_classes=8)
model.load_state_dict(torch.load('final_model.pt'))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['imu_data']  # (batch, 512, 7)
    X = torch.tensor(data).permute(0, 2, 1).float()
    
    with torch.no_grad():
        logits, features = model(X)
        predictions = torch.argmax(logits, dim=1).tolist()
    
    return jsonify({
        'predictions': predictions,
        'probabilities': torch.softmax(logits, dim=1).tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Performance Tuning

### Memory Optimization

```python
# Reduce model size (fewer channels)
model = GlassesNet(input_channels=7, num_classes=8)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")  # ~29K params

# Use mixed precision for faster training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    logits, features = model(X)
    loss = loss_fn(logits, y)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Batch Size Tuning

```python
# Monitor memory usage
import torch

config = Config()
for batch_size in [16, 32, 64, 128, 256]:
    try:
        config.batch_size = batch_size
        train_continual_learning(X_train, y_train, X_test, y_test, config)
        print(f"Batch size {batch_size} OK")
        print(f"GPU memory: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")
    except RuntimeError as e:
        print(f"Batch size {batch_size} failed: {e}")
        break
```

## Monitoring & Logging

### TensorBoard Integration

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./logs')

for batch_idx, (X_batch, y_batch) in enumerate(batches):
    for epoch in range(epochs):
        train_acc = train_epoch(...)
        val_acc = evaluate(...)
        
        writer.add_scalar('train/accuracy', train_acc, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        writer.add_scalar('batch_progress', batch_idx, epoch)

writer.close()
# View: tensorboard --logdir ./logs
```

### Custom Metrics

```python
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score
)

# Per-class metrics
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, predictions, average=None
)

print(f"Per-class F1: {f1}")
print(f"Macro F1: {f1.mean():.3f}")
```

## Troubleshooting

### Issue: Out of Memory (OOM)

```python
# Solution 1: Reduce batch size
config.batch_size = 16  # Instead of 64

# Solution 2: Clear cache
torch.cuda.empty_cache()

# Solution 3: Use CPU instead
config.device = torch.device('cpu')
```

### Issue: Model Not Learning

```python
# Check 1: Data normalization
from train_continual_learning import normalize_input
X_normalized = normalize_input(X_train)
print(f"Mean: {X_normalized.mean()}, Std: {X_normalized.std()}")

# Check 2: Learning rate
config.learning_rate = 0.01  # Try higher

# Check 3: Loss function weights
loss_fn = CrossEntropyLoss(weight=torch.tensor([...]))
```

### Issue: Reproducibility

```python
import random
import torch
import numpy as np

# Set all seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Disable non-deterministic algorithms
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Production Checklist

- [ ] Model training completed and validated
- [ ] Model checkpoint saved (`torch.save(model.state_dict(), path)`)
- [ ] Performance benchmarks recorded
- [ ] Input validation implemented
- [ ] Error handling for edge cases
- [ ] API documentation written
- [ ] Load testing completed
- [ ] Monitoring/logging configured
- [ ] Rollback plan in place
- [ ] Data privacy/security reviewed

## Support

For deployment issues:
1. Check reproducibility settings (seeds, determinism)
2. Verify data format matches expected input shape
3. Test with synthetic data first
4. Profile performance bottlenecks
5. Open GitHub issue with minimal reproduction case
