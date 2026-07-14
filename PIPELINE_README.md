# GlassesNet gait-recognition training pipeline — runnable

Runnable training pipeline for gait recognition from wearable IMU
(accelerometer / gyroscope / magnetometer) sensor windows. A small 1D-CNN
(`GlassesNet`) is used as a universal feature extractor / classifier.

> Data note: the sensor datasets used for this project are **not included** in
> this repository. Point the scripts at your own data directory with the
> `--data-dir` / `--dataset-dir` argument. Each `<subject>.pickle` is expected
> to contain a `"walk"` array of shape `(N, 200, 7)`.

## Results (30 epochs, 10% held-out test split)

| Task | Command | Best val accuracy |
|------|---------|-------------------|
| Binary (2-class) | `numpy_train.py --mode binary` | **97.1%** |
| Multi-class (8 subjects) | `numpy_train.py --mode people` | **94.7%** |

## Two ways to run

**1. `numpy_train.py` — no PyTorch required.**
A faithful NumPy reimplementation of the exact `GlassesNet` architecture,
weighted cross-entropy, Adam, and training loop. Gradients were verified
numerically (max error ~1e-11). Needs only numpy + scikit-learn + matplotlib.

```
python numpy_train.py --mode binary --epochs 30 --dataset-dir path/to/dataset
python numpy_train.py --mode people --epochs 30 --dataset-dir path/to/dataset
```
It checkpoints after every epoch (`<mode>_ckpt.npz`) and auto-resumes, so a run
can be stopped and restarted.

**2. `train_glassesnet_torch.py` — PyTorch version.**
Same model; uses GPU automatically if available.

```
pip install torch scikit-learn seaborn matplotlib pandas tqdm
python train_glassesnet_torch.py --mode binary --data-dir path/to/dataset
```

## Bugs fixed vs. the original training scripts

1. **Hardcoded absolute pickle paths** → real `--data-dir` / `--dataset-dir` arg.
2. **`plt.show()`** (blocks / fails when headless) → figures saved to an output folder.
3. **Label/sample misalignment in the multi-class script** → features and labels
   were built in different subject orders, mislabelling a chunk of samples. Both
   scripts now build `X` and `y` in one per-subject loop, so labels always line up.
4. Class weights are computed with `compute_class_weight("balanced", …)` instead
   of a hardcoded vector.

Each script writes accuracy curves, a normalized confusion matrix, and a metrics
JSON to its output folder.
