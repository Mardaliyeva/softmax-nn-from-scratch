
# -------------------------
# Data loading, shared mathematical functions, evaluation utilities.
# Written by: Ulker
# -------------------------

from __future__ import annotations
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Resolve paths relative to this script so outputs and datasets work no matter where the script is launched from.
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "capstone_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """Set NumPy's global random seed so runs are reproducible."""
    np.random.seed(seed)


set_seed(42)


def locate_file(candidates: list[str]) -> Path:
    """Search a few likely directories and return the first matching dataset file."""
    search_dirs = [
        SCRIPT_DIR,
        Path.cwd(),
        Path("/mnt/data"),
        Path("/mnt/user-data/uploads"),
    ]
    for directory in search_dirs:
        for name in candidates:
            path = directory / name
            if path.exists():
                return path
    raise FileNotFoundError(f"Could not find any of: {candidates}")


def load_np_bundle(path: Path) -> dict[str, np.ndarray]:
    """Load array data from either a plain .npz archive or a .zip file containing .npy arrays."""
    if path.suffix == ".npz":
        data = np.load(path)
        return {k: data[k] for k in data.files}
    if path.suffix == ".zip":
        out: dict[str, np.ndarray] = {}
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                key = Path(name).stem
                with zf.open(name) as f:
                    out[key] = np.load(f)
        return out
    raise ValueError(f"Unsupported file type for {path}")


def load_digits_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the digits dataset plus the predefined train/validation/test split indices."""
    digits_path = locate_file(["digits_data.npz", "digits_data.npz.zip"])
    split_path = locate_file(["digits_split_indices.npz", "digits_split_indices.npz.zip"])

    digits = load_np_bundle(digits_path)
    splits = load_np_bundle(split_path)

    X_all, y_all = digits["X"], digits["y"]
    train_idx = splits["train_idx"]
    val_idx = splits["val_idx"]
    test_idx = splits["test_idx"]

    return (
        X_all[train_idx], y_all[train_idx],
        X_all[val_idx], y_all[val_idx],
        X_all[test_idx], y_all[test_idx],
    )


def load_synthetic_dataset(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load one synthetic dataset by name and return its train/validation/test splits."""
    path = locate_file([f"{name}.npz", f"{name}.npz.zip"])
    data = load_np_bundle(path)
    return (
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        data["X_test"], data["y_test"],
    )


# =============================================================================
# Shared mathematical functions
# =============================================================================


def one_hot(y: np.ndarray, k: int) -> np.ndarray:
    """Convert integer class labels into one-hot encoded rows."""
    Y = np.zeros((len(y), k), dtype=float)
    Y[np.arange(len(y)), y] = 1.0
    return Y


def softmax(S: np.ndarray) -> np.ndarray:
    """Apply a numerically stable softmax to raw class scores."""
    S_shift = S - np.max(S, axis=1, keepdims=True)
    E = np.exp(S_shift)
    return E / np.sum(E, axis=1, keepdims=True)


def cross_entropy(P: np.ndarray, y: np.ndarray) -> float:
    """Compute mean cross-entropy using the probability assigned to the true class."""
    eps = 1e-15
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], eps, 1.0))))


def accuracy(P: np.ndarray, y: np.ndarray) -> float:
    """Compute classification accuracy from predicted probabilities and true labels."""
    return float(np.mean(np.argmax(P, axis=1) == y))


def predictive_entropy(P: np.ndarray) -> np.ndarray:
    """Measure predictive uncertainty for each example using entropy."""
    eps = 1e-15
    return -np.sum(P * np.log(np.clip(P, eps, 1.0)), axis=1)


def compute_ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 5) -> float:
    """Estimate expected calibration error by comparing confidence and empirical accuracy in bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        acc_bin = np.mean(correct[mask])
        conf_bin = np.mean(conf[mask])
        ece += (np.sum(mask) / n) * abs(acc_bin - conf_bin)
    return float(ece)


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 5):
    """Build the per-bin statistics needed to draw a reliability diagram."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centres, emp_acc, counts, avg_conf = [], [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        centres.append((lo + hi) / 2)
        emp_acc.append(np.mean(correct[mask]))
        counts.append(int(np.sum(mask)))
        avg_conf.append(np.mean(conf[mask]))
    return np.array(centres), np.array(emp_acc), np.array(counts), np.array(avg_conf)


def t95_half_width(values: np.ndarray) -> float:
    """Return the half-width of a 95% t-based confidence interval for a small sample."""
    if len(values) < 2:
        return 0.0
    return float(2.776 * values.std(ddof=1) / np.sqrt(len(values)))


def check_probabilities(P: np.ndarray) -> bool:
    """Verify that each probability row sums to 1."""
    return bool(np.allclose(P.sum(axis=1), 1.0, atol=1e-6))


def has_nan_or_inf(arr: np.ndarray) -> bool:
    """Check whether an array contains NaN or infinite values."""
    return bool(np.isnan(arr).any() or np.isinf(arr).any())


# =============================================================================
# Evaluation and training wrappers
# =============================================================================


@dataclass
class EvalResult:
    """Container for the main train, validation, and test metrics of one trained model."""
    train_acc: float
    train_ce: float
    val_acc: float
    val_ce: float
    test_acc: float
    test_ce: float
    best_epoch: int


def evaluate_model(model, X_tr, y_tr, X_val, y_val, X_test, y_test, history) -> EvalResult:
    """Evaluate a trained model on all dataset splits and pack the metrics into an EvalResult."""
    if hasattr(model, "predict_proba"):
        P_tr = model.predict_proba(X_tr)
        P_val = model.predict_proba(X_val)
        P_test = model.predict_proba(X_test)
    else:
        P_tr = model.forward(X_tr)
        P_val = model.forward(X_val)
        P_test = model.forward(X_test)
    return EvalResult(
        train_acc=accuracy(P_tr, y_tr),
        train_ce=cross_entropy(P_tr, y_tr),
        val_acc=accuracy(P_val, y_val),
        val_ce=cross_entropy(P_val, y_val),
        test_acc=accuracy(P_test, y_test),
        test_ce=cross_entropy(P_test, y_test),
        best_epoch=int(history["best_epoch"]),
    )


def train_softmax(
    X_tr, y_tr, X_val, y_val, d, k,
    epochs=200, lr=0.05, lam=1e-4, batch_size=64, seed=42, shuffle_seed=None,
):
    """Helper that constructs, trains, and returns a softmax-regression model plus its history."""
    from aytan_models import SoftmaxRegression
    model = SoftmaxRegression(d=d, k=k, lam=lam, lr=lr, batch_size=batch_size, seed=seed, shuffle_seed=shuffle_seed)
    history = model.train(X_tr, y_tr, X_val, y_val, epochs=epochs)
    return model, history


def train_nn(
    X_tr, y_tr, X_val, y_val, d, h, k,
    epochs=200, lam=1e-4, optimizer="adam", lr=0.001, batch_size=64,
    momentum=0.9, beta1=0.9, beta2=0.999, eps=1e-8,
    seed=42, shuffle_seed=None,
):
    """Helper that constructs, trains, and returns a neural network plus its history."""
    from aytan_models import OneHiddenLayerNet
    model = OneHiddenLayerNet(
        d=d, h=h, k=k, lam=lam, optimizer=optimizer, lr=lr, batch_size=batch_size,
        momentum=momentum, beta1=beta1, beta2=beta2, eps=eps,
        seed=seed, shuffle_seed=shuffle_seed,
    )
    history = model.train(X_tr, y_tr, X_val, y_val, epochs=epochs)
    return model, history


def write_table(path: Path, title: str, rows: list[dict]) -> None:
    """Write a simple fixed-width text table for inclusion in the report outputs."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write("=" * len(title) + "\n\n")
        if not rows:
            f.write("(no rows)\n")
            return
        headers = list(rows[0].keys())
        widths = {h: max(len(h), max(len(f"{r[h]}") for r in rows)) for h in headers}
        f.write(" | ".join(f"{h:<{widths[h]}}" for h in headers) + "\n")
        f.write("-+-".join("-" * widths[h] for h in headers) + "\n")
        for row in rows:
            f.write(" | ".join(f"{str(row[h]):<{widths[h]}}" for h in headers) + "\n")
