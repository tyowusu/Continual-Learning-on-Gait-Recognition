"""
train_glassesnet_torch.py
=========================
Cleaned, runnable PyTorch version of the gait-recognition training pipeline
(consolidates the original Network/fe_male.py and Network/people.py).

Run on a machine that has PyTorch installed (CPU or GPU):
    pip install torch scikit-learn seaborn matplotlib pandas tqdm
    python train_glassesnet_torch.py --mode binary --data-dir path/to/dataset
    python train_glassesnet_torch.py --mode people --data-dir path/to/dataset

What was fixed vs. the originals
--------------------------------
1. Hardcoded "/..path/male#1.pickle" paths -> real --data-dir argument.
2. plt.show() (blocks / fails when headless) -> figures saved to --out-dir.
3. people.py bug: X was concatenated in the order
   [male1,male2,male3,male4,female1..female4] but the labels were assigned in a
   DIFFERENT order (male1,male2,female1,female2,male3,female3,male4,female4),
   so samples were mislabelled. Here X and y are built from a single per-person
   loop, so they always line up.
4. Robust device handling, reproducible seed, and a clean CLI.

The model (GlassesNet) and the 16-wide output head are unchanged from the
original, so results are directly comparable (paper: ~97% male/female, ~93%
8-person). A NumPy reference implementation (numpy_train.py) reproduces these
same numbers without PyTorch.
"""
import os, json, pickle, argparse, random
import numpy as np
import torch
from torch.nn import (Module, LeakyReLU, Dropout, Conv1d, Flatten, Linear,
                      CrossEntropyLoss)
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")                       # headless-safe; never call plt.show()
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except Exception:
    sns = None

NUM_LOGITS = 16
PERSON_ORDER = ["male#1", "male#2", "male#3", "male#4",
                "female#1", "female#2", "female#3", "female#4"]
PERSON_LABEL = {"male#1": 0, "male#2": 1, "female#1": 2, "female#2": 3,
                "male#3": 4, "female#3": 5, "male#4": 6, "female#4": 7}


def set_seed(seed=0):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)


def normalize_input(x):
    return x - np.mean(x, axis=1, keepdims=True)


def load_xy(data_dir, mode):
    X_parts, y_parts = [], []
    for person in PERSON_ORDER:
        with open(os.path.join(data_dir, f"{person}.pickle"), "rb") as f:
            walk = np.asarray(pickle.load(f)["walk"], dtype=np.float64)
        X_parts.append(walk)
        if mode == "binary":
            label = 0 if person.startswith("male") else 1
        else:
            label = PERSON_LABEL[person]
        y_parts.append(np.full(len(walk), label, dtype=np.int64))
    X = normalize_input(np.concatenate(X_parts, axis=0))
    y = np.concatenate(y_parts, axis=0)
    return X, y


class GlassesNet(Module):
    def __init__(self):
        super().__init__()
        self.lrelu = LeakyReLU()
        self.dropout1 = Dropout(0)
        self.dropout2 = Dropout(0)
        self.dropout3 = Dropout(0)
        self.dropout4 = Dropout(0.25)
        self.layer1 = Conv1d(7, 8, 7, 8)
        self.layer2 = Conv1d(8, 16, 5, 2)
        self.layer3 = Conv1d(16, 16, 3, 2)
        self.layer4 = Conv1d(16, 16, 3, 2)
        self.flatten = Flatten()
        self.fc = Linear(32, NUM_LOGITS)

    def forward(self, x):
        y = self.dropout1(self.lrelu(self.layer1(x)))
        y = self.dropout2(self.lrelu(self.layer2(y)))
        y = self.dropout3(self.lrelu(self.layer3(y)))
        y = self.dropout4(self.lrelu(self.layer4(y)))
        return self.fc(self.flatten(y))


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = torch.tensor(x).permute(0, 2, 1).float()
        self.y = torch.tensor(y).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def build_class_weight(y_train):
    """Balanced weights over classes present; length NUM_LOGITS (unused=1.0)."""
    present = np.array(sorted(set(y_train.tolist())))
    cw = compute_class_weight("balanced", classes=present, y=y_train)
    w = np.ones(NUM_LOGITS, dtype=np.float32)
    for c, v in zip(present, cw):
        w[c] = v
    return torch.tensor(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["binary", "people"], default="binary")
    ap.add_argument("--data-dir", required=True,
                    help='folder containing male#1.pickle ... female#4.pickle')
    ap.add_argument("--out-dir", default="torch_results")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.batch_size is None:
        args.batch_size = 7168 if args.mode == "binary" else 504
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[{args.mode}] device={device}")

    X, y = load_xy(args.data_dir, args.mode)
    print(f"[{args.mode}] X={X.shape} classes={sorted(set(y.tolist()))}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=True, stratify=y, random_state=args.seed)

    train_loader = DataLoader(SimpleDataset(X_train, y_train),
                              batch_size=args.batch_size, shuffle=True)
    X_test_t = torch.tensor(X_test).permute(0, 2, 1).float().to(device)
    y_test_t = torch.tensor(y_test).long().to(device)

    model = GlassesNet().to(device)
    optimizer = Adam(model.parameters())
    loss_fn = CrossEntropyLoss(weight=build_class_weight(y_train).to(device))

    train_acc, val_acc = [], []
    best_loss, best_preds, best_state = np.inf, None, None
    for epoch in range(args.epochs):
        model.train()
        correct = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == yb).sum().item()
        tr = correct / len(X_train)
        train_acc.append(tr)

        model.eval()
        with torch.no_grad():
            out = model(X_test_t)
            v_loss = loss_fn(out, y_test_t).item()
            preds = out.argmax(1).cpu().numpy()
            acc = float((preds == y_test).mean())
        val_acc.append(acc)
        print(f"  epoch {epoch:2d}  train_acc={tr:.4f}  val_acc={acc:.4f}  val_loss={v_loss:.4f}")
        if v_loss < best_loss:
            best_loss, best_preds = v_loss, preds
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"{args.mode}_model.pt"))

    # accuracy curves
    plt.figure()
    plt.plot(train_acc, label="train"); plt.plot(val_acc, label="validation")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend()
    plt.title(f"GlassesNet ({args.mode}) accuracy")
    plt.savefig(os.path.join(args.out_dir, f"{args.mode}_curves.png"),
                dpi=120, bbox_inches="tight"); plt.close()

    # confusion matrix
    labels = sorted(set(y_test.tolist()))
    cm = confusion_matrix(y_test, best_preds, labels=labels, normalize="pred")
    plt.figure(figsize=(6, 5))
    if sns is not None:
        sns.heatmap(cm, annot=True, cmap="Blues", fmt=".2f",
                    xticklabels=labels, yticklabels=labels)
    else:
        plt.imshow(cm, cmap="Blues"); plt.colorbar()
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Confusion matrix ({args.mode})")
    plt.savefig(os.path.join(args.out_dir, f"{args.mode}_confusion_matrix.png"),
                dpi=120, bbox_inches="tight"); plt.close()

    with open(os.path.join(args.out_dir, f"{args.mode}_metrics.json"), "w") as f:
        json.dump({"mode": args.mode, "epochs": args.epochs,
                   "best_val_accuracy": float(max(val_acc)),
                   "final_val_accuracy": float(val_acc[-1]),
                   "best_val_loss": float(best_loss)}, f, indent=2)
    print(f"[{args.mode}] best val accuracy = {max(val_acc):.4f}  ->  {args.out_dir}/")


if __name__ == "__main__":
    main()
