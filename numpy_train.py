"""
numpy_train.py
==============
A dependency-light (NumPy + scikit-learn only) reimplementation of the
GlassesNet gait-recognition training pipeline from the thesis repo.

Why this exists: the original pipeline uses PyTorch, which could not be
installed in the execution sandbox (the CPU-only wheel is on a blocked index
and the GPU build's CUDA libraries don't fit the disk). This file reproduces
the *exact* GlassesNet architecture and training loop in pure NumPy so the
pipeline can be validated end-to-end on the real data.

It supports both tasks from the original code:
  --mode binary   -> male vs female   (fe_male.py)
  --mode people   -> 8-person classID (people.py)

Outputs (saved next to the datasets, and echoed to stdout):
  <mode>_curves.png            train/val accuracy per epoch
  <mode>_confusion_matrix.png  normalized confusion matrix on the test split
  <mode>_metrics.json          final accuracy / loss

Run:
  python3 numpy_train.py --mode binary --epochs 30
  python3 numpy_train.py --mode people --epochs 30
"""
import os, sys, json, pickle, argparse, time
import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless: never call plt.show()
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

SEED = 0
np.random.seed(SEED)

# The model always emits 16 logits (matches the original Linear(32, 16)).
NUM_LOGITS = 16

# Per-person identity -> class id for the 8-class task (kept consistent with
# the person -> label intent in the original people.py, but with X and y
# aligned to the SAME person order, fixing the original mismatch bug).
PERSON_LABEL = {
    "male#1": 0, "male#2": 1, "female#1": 2, "female#2": 3,
    "male#3": 4, "female#3": 5, "male#4": 6, "female#4": 7,
}
# Concatenation order used for building X (must match label lookup).
PERSON_ORDER = ["male#1", "male#2", "male#3", "male#4",
                "female#1", "female#2", "female#3", "female#4"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def normalize_input(x):
    """Subtract the per-sample mean over the time axis (axis=1 of (N,200,7))."""
    m = np.mean(x, axis=1, keepdims=True)
    return x - m


def load_dataset(dataset_dir):
    """Load every <person>.pickle and return {person: walk_array(N,200,7)}."""
    walks = {}
    for person in PERSON_ORDER:
        path = os.path.join(dataset_dir, f"{person}.pickle")
        with open(path, "rb") as f:
            d = pickle.load(f)
        walks[person] = np.asarray(d["walk"], dtype=np.float64)
    return walks


def build_xy(walks, mode):
    """Build X (N,200,7) and integer labels y for the requested task."""
    X_parts, y_parts = [], []
    for person in PERSON_ORDER:
        w = walks[person]
        X_parts.append(w)
        if mode == "binary":
            label = 0 if person.startswith("male") else 1
        else:  # people
            label = PERSON_LABEL[person]
        y_parts.append(np.full(len(w), label, dtype=np.int64))
    X = np.concatenate(X_parts, axis=0)
    X = normalize_input(X)
    y = np.concatenate(y_parts, axis=0)
    return X, y


# --------------------------------------------------------------------------- #
# Layers (forward + backward), matching PyTorch defaults
# --------------------------------------------------------------------------- #
def torch_uniform(shape, fan_in):
    """PyTorch default init bound = 1/sqrt(fan_in) for Conv/Linear weights."""
    bound = 1.0 / np.sqrt(fan_in)
    return np.random.uniform(-bound, bound, size=shape).astype(np.float64)


class Conv1d:
    """1D convolution with stride, no padding, dilation=1 (im2col based)."""

    def __init__(self, in_ch, out_ch, k, stride):
        self.in_ch, self.out_ch, self.k, self.stride = in_ch, out_ch, k, stride
        fan_in = in_ch * k
        self.W = torch_uniform((out_ch, in_ch, k), fan_in)
        self.b = torch_uniform((out_ch,), fan_in)
        self.cache = None

    def _im2col(self, x):
        N, C, L = x.shape
        k, s = self.k, self.stride
        L_out = (L - k) // s + 1
        # (N, C, L_out, k)
        cols = np.empty((N, C, L_out, k), dtype=x.dtype)
        for j in range(k):
            cols[:, :, :, j] = x[:, :, j:j + s * L_out:s]
        # (N, C*k, L_out)
        cols = cols.transpose(0, 1, 3, 2).reshape(N, C * k, L_out)
        return cols, L_out

    def forward(self, x):
        N, C, L = x.shape
        cols, L_out = self._im2col(x)                    # (N, C*k, L_out)
        Ck = C * self.k
        Wr = self.W.reshape(self.out_ch, Ck)             # (O, C*k)
        # BLAS gemm: (O,Ck) @ (Ck, N*L_out) -> (O, N*L_out)
        cols2 = cols.transpose(1, 0, 2).reshape(Ck, N * L_out)
        out = (Wr @ cols2).reshape(self.out_ch, N, L_out).transpose(1, 0, 2)
        out = out + self.b[None, :, None]
        self.cache = (x.shape, cols2, L_out, Ck)
        return out

    def backward(self, dout):
        x_shape, cols2, L_out, Ck = self.cache
        N, C, L = x_shape
        k, s = self.k, self.stride
        O = self.out_ch
        Wr = self.W.reshape(O, Ck)
        dout2 = dout.transpose(1, 0, 2).reshape(O, N * L_out)   # (O, N*L_out)
        self.db = dout.sum(axis=(0, 2))
        self.dW = (dout2 @ cols2.T).reshape(self.W.shape)        # (O, Ck)
        dcols2 = Wr.T @ dout2                                    # (Ck, N*L_out)
        dcols = dcols2.reshape(Ck, N, L_out).transpose(1, 0, 2)  # (N, Ck, L_out)
        dcols = dcols.reshape(N, C, k, L_out).transpose(0, 1, 3, 2)  # (N,C,L_out,k)
        dx = np.zeros(x_shape, dtype=dout.dtype)
        for j in range(k):
            dx[:, :, j:j + s * L_out:s] += dcols[:, :, :, j]
        return dx


class LeakyReLU:
    def __init__(self, slope=0.01):
        self.slope = slope
        self.mask = None

    def forward(self, x):
        self.mask = np.where(x >= 0, 1.0, self.slope)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class Dropout:
    def __init__(self, p):
        self.p = p
        self.mask = None

    def forward(self, x, train):
        if not train or self.p == 0:
            self.mask = None
            return x
        keep = 1.0 - self.p
        self.mask = (np.random.rand(*x.shape) < keep) / keep  # inverted dropout
        return x * self.mask

    def backward(self, dout):
        return dout if self.mask is None else dout * self.mask


class Linear:
    def __init__(self, in_f, out_f):
        self.W = torch_uniform((out_f, in_f), in_f)   # (out, in)
        self.b = torch_uniform((out_f,), in_f)
        self.cache = None

    def forward(self, x):
        self.cache = x
        return x @ self.W.T + self.b

    def backward(self, dout):
        x = self.cache
        self.dW = dout.T @ x
        self.db = dout.sum(axis=0)
        return dout @ self.W


class GlassesNet:
    """Conv1d(7,8,7,8) -> Conv1d(8,16,5,2) -> Conv1d(16,16,3,2)
       -> Conv1d(16,16,3,2) -> Flatten -> Linear(32,16), LeakyReLU between,
       Dropout(0.25) before flatten."""

    def __init__(self):
        self.c1 = Conv1d(7, 8, 7, 8)
        self.c2 = Conv1d(8, 16, 5, 2)
        self.c3 = Conv1d(16, 16, 3, 2)
        self.c4 = Conv1d(16, 16, 3, 2)
        self.a = [LeakyReLU(), LeakyReLU(), LeakyReLU(), LeakyReLU()]
        self.d4 = Dropout(0.25)
        self.fc = Linear(32, NUM_LOGITS)
        self.params = [self.c1, self.c2, self.c3, self.c4, self.fc]

    def forward(self, x, train):
        y = self.a[0].forward(self.c1.forward(x))
        y = self.a[1].forward(self.c2.forward(y))
        y = self.a[2].forward(self.c3.forward(y))
        y = self.a[3].forward(self.c4.forward(y))
        y = self.d4.forward(y, train)
        self._flat_shape = y.shape
        y = y.reshape(y.shape[0], -1)
        return self.fc.forward(y)

    def backward(self, dlogits):
        d = self.fc.backward(dlogits)
        d = d.reshape(self._flat_shape)
        d = self.d4.backward(d)
        d = self.c4.backward(self.a[3].backward(d))
        d = self.c3.backward(self.a[2].backward(d))
        d = self.c2.backward(self.a[1].backward(d))
        d = self.c1.backward(self.a[0].backward(d))
        return d


# --------------------------------------------------------------------------- #
# Loss + optimizer
# --------------------------------------------------------------------------- #
def softmax_ce_weighted(logits, y, weight):
    """Weighted softmax cross-entropy (matches nn.CrossEntropyLoss(weight=...)).
    Returns (mean_loss, dlogits)."""
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    p = ez / ez.sum(axis=1, keepdims=True)
    n = logits.shape[0]
    w = weight[y]                                   # per-sample weight
    logp = np.log(p[np.arange(n), y] + 1e-12)
    loss = -(w * logp).sum() / w.sum()              # torch normalizes by sum(w)
    dl = p.copy()
    dl[np.arange(n), y] -= 1.0
    dl *= w[:, None]
    dl /= w.sum()
    return loss, dl


class Adam:
    def __init__(self, layers, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0
        self.slots = []
        for layer in layers:
            self.slots.append({
                "mW": np.zeros_like(layer.W), "vW": np.zeros_like(layer.W),
                "mb": np.zeros_like(layer.b), "vb": np.zeros_like(layer.b),
            })
        self.layers = layers

    def step(self):
        self.t += 1
        for layer, s in zip(self.layers, self.slots):
            for p, g, mk, vk in [("W", "dW", "mW", "vW"), ("b", "db", "mb", "vb")]:
                grad = getattr(layer, g)
                s[mk] = self.b1 * s[mk] + (1 - self.b1) * grad
                s[vk] = self.b2 * s[vk] + (1 - self.b2) * grad ** 2
                mhat = s[mk] / (1 - self.b1 ** self.t)
                vhat = s[vk] / (1 - self.b2 ** self.t)
                setattr(layer, p, getattr(layer, p) - self.lr * mhat / (np.sqrt(vhat) + self.eps))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def make_weight_vector(y_train, present_classes):
    """Balanced class weights over the classes that actually appear; the unused
    logits keep weight 1.0 (they never appear as a target)."""
    w = np.ones(NUM_LOGITS, dtype=np.float64)
    cw = compute_class_weight("balanced", classes=present_classes, y=y_train)
    for c, val in zip(present_classes, cw):
        w[c] = val
    return w


def _load_xy_cached(mode, dataset_dir, out_dir):
    """Load X,y with a small .npy cache so repeated runs start fast."""
    xc = os.path.join(out_dir, "X_all.npy")
    yc = os.path.join(out_dir, f"y_{mode}.npy")
    if os.path.exists(xc) and os.path.exists(yc):
        return np.load(xc), np.load(yc)
    walks = load_dataset(dataset_dir)
    X, y = build_xy(walks, mode)
    os.makedirs(out_dir, exist_ok=True)
    np.save(xc, X); np.save(yc, y)
    return X, y


def _save_ckpt(path, net, opt, ep, train_acc, val_acc, best_loss, best_preds):
    blob = {"epoch": ep, "t": opt.t, "train_acc": np.array(train_acc),
            "val_acc": np.array(val_acc), "best_loss": best_loss,
            "best_preds": np.array([] if best_preds is None else best_preds)}
    for i, L in enumerate(net.params):
        blob[f"W{i}"] = L.W; blob[f"b{i}"] = L.b
        blob[f"mW{i}"] = opt.slots[i]["mW"]; blob[f"vW{i}"] = opt.slots[i]["vW"]
        blob[f"mb{i}"] = opt.slots[i]["mb"]; blob[f"vb{i}"] = opt.slots[i]["vb"]
    tmp = path + ".tmp.npz"          # atomic write: temp then rename
    np.savez(tmp, **blob)
    os.replace(tmp, path)


def _load_ckpt(path, net, opt):
    d = np.load(path, allow_pickle=True)
    for i, L in enumerate(net.params):
        L.W = d[f"W{i}"]; L.b = d[f"b{i}"]
        opt.slots[i]["mW"] = d[f"mW{i}"]; opt.slots[i]["vW"] = d[f"vW{i}"]
        opt.slots[i]["mb"] = d[f"mb{i}"]; opt.slots[i]["vb"] = d[f"vb{i}"]
    opt.t = int(d["t"])
    bp = d["best_preds"]
    return (int(d["epoch"]), list(d["train_acc"]), list(d["val_acc"]),
            float(d["best_loss"]), (None if bp.size == 0 else bp))


def train(mode, dataset_dir, epochs, batch_size, out_dir, resume=True):
    X, y = _load_xy_cached(mode, dataset_dir, out_dir)
    print(f"[{mode}] X={X.shape} y={y.shape} classes={sorted(set(y.tolist()))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=True, stratify=y, random_state=SEED)

    # (N,200,7) -> (N,7,200)
    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    present = np.array(sorted(set(y.tolist())))
    weight = make_weight_vector(y_train, present)

    net = GlassesNet()
    opt = Adam(net.params)

    n = len(X_train)
    train_acc, val_acc = [], []
    best_loss, best_preds = np.inf, None

    ckpt = os.path.join(out_dir, f"{mode}_ckpt.npz")
    start_ep = 0
    if resume and os.path.exists(ckpt):
        start_ep, train_acc, val_acc, best_loss, best_preds = _load_ckpt(ckpt, net, opt)
        print(f"[{mode}] resumed from epoch {start_ep}")

    for ep in range(start_ep, epochs):
        idx = np.random.permutation(n)
        correct = 0
        for start in range(0, n, batch_size):
            bi = idx[start:start + batch_size]
            xb, yb = X_train[bi], y_train[bi]
            logits = net.forward(xb, train=True)
            loss, dl = softmax_ce_weighted(logits, yb, weight)
            net.backward(dl)
            opt.step()
            correct += (logits.argmax(1) == yb).sum()
        tr = correct / n
        train_acc.append(tr)

        logits = net.forward(X_test, train=False)
        val, dl = softmax_ce_weighted(logits, y_test, weight)
        preds = logits.argmax(1)
        acc = (preds == y_test).mean()
        val_acc.append(acc)
        print(f"  epoch {ep:2d}  train_acc={tr:.4f}  val_acc={acc:.4f}  val_loss={val:.4f}")
        if val < best_loss:
            best_loss, best_preds = val, preds
        _save_ckpt(ckpt, net, opt, ep + 1, train_acc, val_acc, best_loss, best_preds)

    # ---- outputs ----
    os.makedirs(out_dir, exist_ok=True)
    # curves
    plt.figure()
    plt.plot(train_acc, label="train")
    plt.plot(val_acc, label="validation")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend()
    plt.title(f"GlassesNet ({mode}) accuracy")
    plt.savefig(os.path.join(out_dir, f"{mode}_curves.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # confusion matrix
    labels = sorted(set(y_test.tolist()))
    cm = confusion_matrix(y_test, best_preds, labels=labels, normalize="pred")
    plt.figure(figsize=(6, 5))
    try:
        import seaborn as sns
        sns.heatmap(cm, annot=True, cmap="Blues", fmt=".2f",
                    xticklabels=labels, yticklabels=labels)
    except Exception:
        plt.imshow(cm, cmap="Blues"); plt.colorbar()
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Confusion matrix ({mode})")
    plt.savefig(os.path.join(out_dir, f"{mode}_confusion_matrix.png"), dpi=120, bbox_inches="tight")
    plt.close()

    metrics = {
        "mode": mode, "epochs": epochs, "batch_size": batch_size,
        "n_train": int(n), "n_test": int(len(X_test)),
        "best_val_accuracy": float(max(val_acc)),
        "final_val_accuracy": float(val_acc[-1]),
        "best_val_loss": float(best_loss),
    }
    with open(os.path.join(out_dir, f"{mode}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{mode}] best val accuracy = {max(val_acc):.4f}")
    return metrics


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    default_ds = os.path.join(here, "data")  # folder with male#1.pickle ... female#4.pickle
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["binary", "people"], default="binary")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--dataset-dir", default=default_ds)
    ap.add_argument("--out-dir", default=here)
    args = ap.parse_args()
    if args.batch_size is None:
        args.batch_size = 7168 if args.mode == "binary" else 504
    t0 = time.time()
    train(args.mode, args.dataset_dir, args.epochs, args.batch_size, args.out_dir)
    print(f"done in {time.time() - t0:.1f}s")
