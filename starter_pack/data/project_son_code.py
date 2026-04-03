# Capstone experiment suite
# -------------------------
# This script loads datasets, trains a softmax regression model and a one-hidden-layer
# neural network, runs the required experiments, saves plots/tables, and writes summary metrics.
#
from __future__ import annotations
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


# Resolve paths relative to this script so outputs and datasets work no matter where the script is launched from.
SCRIPT_DIR = Path(__file__).resolve().parent
# Store every figure, table, and JSON summary in one dedicated output folder.
OUTPUT_DIR = SCRIPT_DIR / "capstone_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    """Set NumPy's global random seed so runs are reproducible."""
    np.random.seed(seed)


# Fix the default global seed once at import time for reproducible behavior.
set_seed(42)



def locate_file(candidates: list[str]) -> Path:
    """Search a few likely directories and return the first matching dataset file."""
    # Try a few common locations where the datasets may have been uploaded or extracted.
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
    """Load either an .npz file or a .zip file that contains .npy arrays."""
    # Plain NPZ archives can be loaded directly by NumPy.
    if path.suffix == ".npz":
        data = np.load(path)
        return {k: data[k] for k in data.files}
    # Some datasets are packaged as ZIP files containing individual NPY arrays.
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
# Utilities
# =============================================================================


def one_hot(y: np.ndarray, k: int) -> np.ndarray:
    """Convert integer class labels into one-hot encoded rows."""
    Y = np.zeros((len(y), k), dtype=float)
    # Put a 1 in the column of the true class for each sample.
    Y[np.arange(len(y)), y] = 1.0
    return Y



def softmax(S: np.ndarray) -> np.ndarray:
    """Apply a numerically stable softmax to raw class scores."""
    # Subtract the row-wise maximum to avoid numerical overflow in exp().
    S_shift = S - np.max(S, axis=1, keepdims=True)
    E = np.exp(S_shift)
    return E / np.sum(E, axis=1, keepdims=True)



def cross_entropy(P: np.ndarray, y: np.ndarray) -> float:
    """Compute mean cross-entropy using the probability assigned to the true class."""
    eps = 1e-15
    # Clip probabilities away from zero so log() stays finite.
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
    # Process one confidence bin at a time.
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
    # Process one confidence bin at a time.
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
# Models
# =============================================================================


class SoftmaxRegression:
    """Multiclass linear classifier trained with mini-batch gradient descent and L2 regularization."""
    def __init__(
        self,
        d: int,
        k: int,
        lam: float = 1e-4,
        lr: float = 0.05,
        batch_size: int = 64,
        seed: int = 42,
        shuffle_seed: int | None = None,
    ):
        """Initialize model parameters and training hyperparameters."""
        rng = np.random.default_rng(seed)
        # Small random initialization is enough for a linear softmax model.
        self.W = rng.normal(0.0, 0.01, size=(k, d))
        self.b = np.zeros(k)
        self.k = k
        self.lam = lam
        self.lr = lr
        self.bs = batch_size
        self.shuffle_seed = seed if shuffle_seed is None else shuffle_seed

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities for a batch of inputs."""
        return softmax(X @ self.W.T + self.b)

    def loss(self, X: np.ndarray, y: np.ndarray, include_reg: bool = True) -> float:
        """Compute data loss and, optionally, the L2 regularization term."""
        P = self.forward(X)
        ce = cross_entropy(P, y)
        if not include_reg:
            return ce
        reg = 0.5 * self.lam * np.sum(self.W ** 2)
        return float(ce + reg)

    def gradients(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute analytical gradients of the softmax loss with respect to weights and biases."""
        n = len(y)
        P = self.forward(X)
        Y = one_hot(y, self.k)
        # For softmax plus cross-entropy, the score gradient simplifies to P - Y.
        dS = (P - Y) / n
        dW = dS.T @ X + self.lam * self.W
        db = dS.sum(axis=0)
        return dW, db

    def step(self, X: np.ndarray, y: np.ndarray) -> None:
        """Perform one gradient-descent parameter update on the current mini-batch."""
        dW, db = self.gradients(X, y)
        self.W -= self.lr * dW
        self.b -= self.lr * db

    def train(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 200,
    ) -> dict[str, list[float] | int]:
        """Train for a fixed number of epochs and keep the parameters with the best validation cross-entropy."""
        rng = np.random.default_rng(self.shuffle_seed)
        n = len(y_tr)
        best_val_ce = np.inf
        best_params = (self.W.copy(), self.b.copy())
        best_epoch = 0
        history: dict[str, list[float] | int] = {
            "train_ce": [], "val_ce": [], "train_acc": [], "val_acc": [], "best_epoch": 0
        }
        for ep in range(epochs):
            # Shuffle the training data at the start of every epoch.
            idx = rng.permutation(n)
            for i in range(0, n, self.bs):
                batch = idx[i:i + self.bs]
                self.step(X_tr[batch], y_tr[batch])

            P_tr = self.forward(X_tr)
            P_val = self.forward(X_val)
            tr_ce = cross_entropy(P_tr, y_tr)
            val_ce = cross_entropy(P_val, y_val)
            tr_acc = accuracy(P_tr, y_tr)
            val_acc = accuracy(P_val, y_val)

            history["train_ce"].append(tr_ce)
            history["val_ce"].append(val_ce)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(val_acc)

            # Keep the checkpoint that gives the best validation cross-entropy.
            if val_ce < best_val_ce:
                best_val_ce = val_ce
                best_params = (self.W.copy(), self.b.copy())
                best_epoch = ep + 1

        self.W, self.b = best_params
        history["best_epoch"] = best_epoch
        return history


class OneHiddenLayerNet:
    """Single-hidden-layer neural network with tanh activations and support for SGD, Momentum, and Adam."""
    def __init__(
        self,
        d: int,
        h: int,
        k: int,
        lam: float = 1e-4,
        optimizer: str = "sgd",
        lr: float = 0.05,
        momentum: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        batch_size: int = 64,
        seed: int = 42,
        shuffle_seed: int | None = None,
    ):
        """Initialize network weights, biases, optimizer state, and training hyperparameters."""
        rng = np.random.default_rng(seed)
        self.h = h
        self.k = k
        # He-style scaling keeps activations in a reasonable range at initialization.
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / d), size=(h, d))
        self.b1 = np.zeros(h)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / h), size=(k, h))
        self.b2 = np.zeros(k)

        self.lam = lam
        self.optimizer = optimizer.lower()
        self.lr = lr
        self.momentum_coef = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps_adam = eps
        self.bs = batch_size
        self.shuffle_seed = seed if shuffle_seed is None else shuffle_seed

        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)
        self.mW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2)
        self.sW1 = np.zeros_like(self.W1)
        self.sb1 = np.zeros_like(self.b1)
        self.sW2 = np.zeros_like(self.W2)
        self.sb2 = np.zeros_like(self.b2)
        self.t = 0

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run the full forward pass and return intermediate activations for backpropagation."""
        Z1 = X @ self.W1.T + self.b1
        # The hidden layer uses tanh to learn a nonlinear feature mapping.
        H = np.tanh(Z1)
        S = H @ self.W2.T + self.b2
        P = softmax(S)
        return Z1, H, S, P

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Convenience wrapper that returns only the predicted class probabilities."""
        return self.forward(X)[-1]

    def loss(self, X: np.ndarray, y: np.ndarray, include_reg: bool = True) -> float:
        """Compute cross-entropy loss plus optional L2 regularization for both weight matrices."""
        P = self.predict_proba(X)
        ce = cross_entropy(P, y)
        if not include_reg:
            return ce
        reg = 0.5 * self.lam * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return float(ce + reg)

    def gradients(self, X: np.ndarray, y: np.ndarray):
        """Backpropagate through the network to obtain gradients for all parameters."""
        n = len(y)
        _, H, _, P = self.forward(X)
        Y = one_hot(y, self.k)
        # For softmax plus cross-entropy, the score gradient simplifies to P - Y.
        dS = (P - Y) / n
        dW2 = dS.T @ H + self.lam * self.W2
        db2 = dS.sum(axis=0)
        dH = dS @ self.W2
        # Derivative of tanh(z) is 1 - tanh(z)^2.
        dZ1 = dH * (1.0 - H ** 2)
        dW1 = dZ1.T @ X + self.lam * self.W1
        db1 = dZ1.sum(axis=0)
        return dW1, db1, dW2, db2

    def _update_parameter(
        self,
        param: np.ndarray,
        grad: np.ndarray,
        v_buf: np.ndarray,
        m_buf: np.ndarray,
        s_buf: np.ndarray,
    ):
        """Update one parameter tensor using the selected optimizer rule."""
        # Plain stochastic gradient descent update.
        if self.optimizer == "sgd":
            param = param - self.lr * grad
            return param, v_buf, m_buf, s_buf
        # Momentum accumulates a velocity to smooth noisy gradient directions.
        if self.optimizer == "momentum":
            v_new = self.momentum_coef * v_buf + self.lr * grad
            param = param - v_new
            return param, v_new, m_buf, s_buf
        # Adam maintains first and second moment estimates for adaptive step sizes.
        if self.optimizer == "adam":
            m_new = self.beta1 * m_buf + (1.0 - self.beta1) * grad
            s_new = self.beta2 * s_buf + (1.0 - self.beta2) * (grad ** 2)
            m_hat = m_new / (1.0 - self.beta1 ** self.t)
            s_hat = s_new / (1.0 - self.beta2 ** self.t)
            param = param - self.lr * m_hat / (np.sqrt(s_hat) + self.eps_adam)
            return param, v_buf, m_new, s_new
        raise ValueError(f"Unknown optimizer: {self.optimizer}")

    def step(self, X: np.ndarray, y: np.ndarray) -> None:
        """Take one optimization step on a mini-batch and update the optimizer timestep."""
        self.t += 1
        dW1, db1, dW2, db2 = self.gradients(X, y)
        self.W1, self.vW1, self.mW1, self.sW1 = self._update_parameter(self.W1, dW1, self.vW1, self.mW1, self.sW1)
        self.b1, self.vb1, self.mb1, self.sb1 = self._update_parameter(self.b1, db1, self.vb1, self.mb1, self.sb1)
        self.W2, self.vW2, self.mW2, self.sW2 = self._update_parameter(self.W2, dW2, self.vW2, self.mW2, self.sW2)
        self.b2, self.vb2, self.mb2, self.sb2 = self._update_parameter(self.b2, db2, self.vb2, self.mb2, self.sb2)

    def train(
        self,
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 200,
    ) -> dict[str, list[float] | int]:
        """Train the neural network and restore the checkpoint with the best validation cross-entropy."""
        rng = np.random.default_rng(self.shuffle_seed)
        n = len(y_tr)
        best_val_ce = np.inf
        best_epoch = 0
        best_params = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())
        history: dict[str, list[float] | int] = {
            "train_ce": [], "val_ce": [], "train_acc": [], "val_acc": [], "best_epoch": 0
        }
        for ep in range(epochs):
            # Shuffle the training data at the start of every epoch.
            idx = rng.permutation(n)
            for i in range(0, n, self.bs):
                batch = idx[i:i + self.bs]
                self.step(X_tr[batch], y_tr[batch])

            P_tr = self.predict_proba(X_tr)
            P_val = self.predict_proba(X_val)
            tr_ce = cross_entropy(P_tr, y_tr)
            val_ce = cross_entropy(P_val, y_val)
            tr_acc = accuracy(P_tr, y_tr)
            val_acc = accuracy(P_val, y_val)

            history["train_ce"].append(tr_ce)
            history["val_ce"].append(val_ce)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(val_acc)

            # Keep the checkpoint that gives the best validation cross-entropy.
            if val_ce < best_val_ce:
                best_val_ce = val_ce
                best_epoch = ep + 1
                best_params = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())

        self.W1, self.b1, self.W2, self.b2 = best_params
        history["best_epoch"] = best_epoch
        return history


# =============================================================================
# Sanity checks
# =============================================================================


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Compute a stable relative error used when comparing analytical and numerical gradients."""
    return float(np.max(np.abs(a - b) / np.maximum(1e-12, np.abs(a) + np.abs(b))))



def finite_difference_softmax(model: SoftmaxRegression, X: np.ndarray, y: np.ndarray, num_checks: int = 6) -> float:
    """Check the softmax model gradients against finite-difference approximations."""
    rng = np.random.default_rng(123)
    dW, db = model.gradients(X, y)
    max_err = 0.0
    eps = 1e-5

    for _ in range(num_checks):
        i = rng.integers(model.W.shape[0])
        j = rng.integers(model.W.shape[1])
        old = model.W[i, j]
        model.W[i, j] = old + eps
        lp = model.loss(X, y, include_reg=True)
        model.W[i, j] = old - eps
        lm = model.loss(X, y, include_reg=True)
        model.W[i, j] = old
        num = (lp - lm) / (2 * eps)
        max_err = max(max_err, abs(num - dW[i, j]) / max(1e-12, abs(num) + abs(dW[i, j])))

    for _ in range(min(2, len(db))):
        i = rng.integers(len(model.b))
        old = model.b[i]
        model.b[i] = old + eps
        lp = model.loss(X, y, include_reg=True)
        model.b[i] = old - eps
        lm = model.loss(X, y, include_reg=True)
        model.b[i] = old
        num = (lp - lm) / (2 * eps)
        max_err = max(max_err, abs(num - db[i]) / max(1e-12, abs(num) + abs(db[i])))

    return float(max_err)



def finite_difference_nn(model: OneHiddenLayerNet, X: np.ndarray, y: np.ndarray, num_checks: int = 8) -> float:
    """Check the neural-network gradients against finite-difference approximations."""
    rng = np.random.default_rng(456)
    dW1, db1, dW2, db2 = model.gradients(X, y)
    eps = 1e-5
    max_err = 0.0

    def check_param(array: np.ndarray, grad: np.ndarray):
        """Numerically test a random subset of entries from one parameter tensor."""
        nonlocal max_err
        for _ in range(num_checks // 4):
            idx = tuple(rng.integers(s) for s in array.shape)
            old = array[idx]
            array[idx] = old + eps
            lp = model.loss(X, y, include_reg=True)
            array[idx] = old - eps
            lm = model.loss(X, y, include_reg=True)
            array[idx] = old
            num = (lp - lm) / (2 * eps)
            ana = grad[idx]
            err = abs(num - ana) / max(1e-12, abs(num) + abs(ana))
            max_err = max(max_err, err)

    check_param(model.W1, dW1)
    check_param(model.W2, dW2)

    for _ in range(2):
        i = rng.integers(len(model.b1))
        old = model.b1[i]
        model.b1[i] = old + eps
        lp = model.loss(X, y, include_reg=True)
        model.b1[i] = old - eps
        lm = model.loss(X, y, include_reg=True)
        model.b1[i] = old
        num = (lp - lm) / (2 * eps)
        ana = db1[i]
        max_err = max(max_err, abs(num - ana) / max(1e-12, abs(num) + abs(ana)))

    for _ in range(2):
        i = rng.integers(len(model.b2))
        old = model.b2[i]
        model.b2[i] = old + eps
        lp = model.loss(X, y, include_reg=True)
        model.b2[i] = old - eps
        lm = model.loss(X, y, include_reg=True)
        model.b2[i] = old
        num = (lp - lm) / (2 * eps)
        ana = db2[i]
        max_err = max(max_err, abs(num - ana) / max(1e-12, abs(num) + abs(ana)))

    return float(max_err)



def overfit_small_batch(model, X: np.ndarray, y: np.ndarray, epochs: int = 400) -> float:
    """Sanity check: confirm that a model can memorize a tiny batch."""
    X_small = X[:20]
    y_small = y[:20]
    model.train(X_small, y_small, X_small, y_small, epochs=epochs)
    P = model.predict_proba(X_small) if hasattr(model, "predict_proba") else model.forward(X_small)
    return accuracy(P, y_small)


# =============================================================================
# Experiment helpers
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


# =============================================================================
# Plotting helpers
# =============================================================================


BLUE = "#2563EB"
RED = "#DC2626"
GREEN = "#16A34A"
GRAY = "#6B7280"
AMBER = "#D97706"
PURPLE = "#7C3AED"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})



def _predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Uniform prediction helper so plotting code works for either model class."""
    return model.predict_proba(X) if hasattr(model, "predict_proba") else model.forward(X)



def plot_binary_boundary(ax, model, X, y, title: str, misclassified: np.ndarray | None = None) -> None:
    """Plot a 2D decision boundary together with the dataset and optional misclassified points."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    # Flatten the coordinate grid into a batch of 2D points for prediction.
    grid = np.c_[xx.ravel(), yy.ravel()]
    P = _predict_proba(model, grid)
    Z = P[:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), cmap="RdBu", alpha=0.32)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1.2)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=22, c=BLUE, edgecolor="white", linewidth=0.5, label="Class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=22, c=RED, edgecolor="white", linewidth=0.5, label="Class 1")
    if misclassified is not None and np.any(misclassified):
        ax.scatter(
            X[misclassified, 0], X[misclassified, 1],
            s=90, facecolors="none", edgecolors="black", linewidths=1.4, label="Misclassified"
        )
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")



def save_training_dynamics(histories: dict[str, dict], outpath: Path, title: str) -> None:
    """Save side-by-side plots of cross-entropy and accuracy across training epochs."""
    # Plot cross-entropy and accuracy side by side for easier comparison.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    color_cycle = [BLUE, RED, GREEN, PURPLE, AMBER]
    for (label, hist), color in zip(histories.items(), color_cycle):
        ep = np.arange(1, len(hist["train_ce"]) + 1)
        axes[0].plot(ep, hist["train_ce"], color=color, lw=1.8, label=f"{label} train")
        axes[0].plot(ep, hist["val_ce"], color=color, lw=1.8, ls="--", label=f"{label} val")
        axes[1].plot(ep, hist["train_acc"], color=color, lw=1.8, label=f"{label} train")
        axes[1].plot(ep, hist["val_acc"], color=color, lw=1.8, ls="--", label=f"{label} val")
    axes[0].set_title("Cross-entropy")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(ncol=2)
    axes[0].set_ylabel("CE")
    axes[1].set_ylabel("Accuracy")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Synthetic tasks
# =============================================================================



def run_synthetic_core_experiment(dataset_name: str, nn_width: int = 32) -> dict:
    """Train both models on one synthetic task, evaluate them, and save a decision-boundary comparison."""
    X_tr, y_tr, X_val, y_val, X_test, y_test = load_synthetic_dataset(dataset_name)
    d = X_tr.shape[1]
    k = len(np.unique(y_tr))

    # Train the linear baseline first.
    softmax_model, softmax_hist = train_softmax(
        X_tr, y_tr, X_val, y_val, d=d, k=k,
        epochs=300, lr=0.05, lam=1e-4, batch_size=32, seed=42, shuffle_seed=42,
    )
    # Then train the nonlinear neural-network model on the same split.
    nn_model, nn_hist = train_nn(
        X_tr, y_tr, X_val, y_val, d=d, h=nn_width, k=k,
        epochs=300, lam=1e-4, optimizer="adam", lr=0.01, batch_size=32,
        seed=42, shuffle_seed=42,
    )

    softmax_eval = evaluate_model(softmax_model, X_tr, y_tr, X_val, y_val, X_test, y_test, softmax_hist)
    nn_eval = evaluate_model(nn_model, X_tr, y_tr, X_val, y_val, X_test, y_test, nn_hist)

    all_X = np.vstack([X_tr, X_val, X_test])
    all_y = np.concatenate([y_tr, y_val, y_test])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    fig.suptitle(f"{dataset_name.replace('_', ' ').title()} · Decision-boundary comparison", fontsize=13, fontweight="bold")
    plot_binary_boundary(axes[0], softmax_model, all_X, all_y, f"Softmax Reg\nTest acc={softmax_eval.test_acc:.3f}, CE={softmax_eval.test_ce:.3f}")
    plot_binary_boundary(axes[1], nn_model, all_X, all_y, f"NN (h={nn_width})\nTest acc={nn_eval.test_acc:.3f}, CE={nn_eval.test_ce:.3f}")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{dataset_name}_decision_boundaries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "dataset": dataset_name,
        "softmax_eval": softmax_eval,
        "nn_eval": nn_eval,
        "softmax_model": softmax_model,
        "nn_model": nn_model,
        "softmax_history": softmax_hist,
        "nn_history": nn_hist,
        "all_X": all_X,
        "all_y": all_y,
        "X_test": X_test,
        "y_test": y_test,
    }



def run_moons_capacity_ablation() -> dict:
    """Compare multiple hidden-layer widths on the moons dataset to study model capacity."""
    X_tr, y_tr, X_val, y_val, X_test, y_test = load_synthetic_dataset("moons")
    d = X_tr.shape[1]
    k = len(np.unique(y_tr))
    # Small, medium, and larger hidden widths let us study underfitting versus adequate capacity.
    widths = [2, 8, 32]
    results = []
    trained_models = {}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    fig.suptitle("Moons · Capacity ablation (hidden widths 2, 8, 32)", fontsize=13, fontweight="bold")
    all_X = np.vstack([X_tr, X_val, X_test])
    all_y = np.concatenate([y_tr, y_val, y_test])

    for ax, h in zip(axes, widths):
        model, hist = train_nn(
            X_tr, y_tr, X_val, y_val, d=d, h=h, k=k,
            epochs=300, lam=1e-4, optimizer="adam", lr=0.01, batch_size=32,
            seed=42, shuffle_seed=42,
        )
        ev = evaluate_model(model, X_tr, y_tr, X_val, y_val, X_test, y_test, hist)
        plot_binary_boundary(ax, model, all_X, all_y, f"h={h}\nTest acc={ev.test_acc:.3f}, CE={ev.test_ce:.3f}")
        trained_models[h] = (model, hist, ev)
        results.append({
            "width": h,
            "best_epoch": ev.best_epoch,
            "val_acc": round(ev.val_acc, 4),
            "val_ce": round(ev.val_ce, 4),
            "test_acc": round(ev.test_acc, 4),
            "test_ce": round(ev.test_ce, 4),
        })

    axes[-1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "moons_capacity_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    write_table(OUTPUT_DIR / "moons_capacity_ablation_table.txt", "Moons capacity ablation", results)
    return {"rows": results, "trained_models": trained_models, "all_X": all_X, "all_y": all_y, "X_test": X_test, "y_test": y_test}



def run_failure_case_analysis(moons_ablation: dict) -> dict:
    """Visualize and explain how an under-capacity network fails on the moons dataset."""
    # Contrast the most under-parameterized model against the strongest capacity setting.
    model_2, hist_2, ev_2 = moons_ablation["trained_models"][2]
    model_32, hist_32, ev_32 = moons_ablation["trained_models"][32]
    X_test = moons_ablation["X_test"]
    y_test = moons_ablation["y_test"]
    all_X = moons_ablation["all_X"]
    all_y = moons_ablation["all_y"]

    pred_2 = np.argmax(model_2.predict_proba(X_test), axis=1)
    pred_32 = np.argmax(model_32.predict_proba(X_test), axis=1)
    mis_2 = pred_2 != y_test
    mis_32 = pred_32 != y_test

    fig = plt.figure(figsize=(11, 4.5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    fig.suptitle("Failure case · under-capacity on moons (h=2)", fontsize=13, fontweight="bold")

    plot_binary_boundary(ax1, model_2, all_X, all_y, f"Failure model: h=2\nTest acc={ev_2.test_acc:.3f}, CE={ev_2.test_ce:.3f}")
    plot_binary_boundary(ax2, model_32, all_X, all_y, f"Reference model: h=32\nTest acc={ev_32.test_acc:.3f}, CE={ev_32.test_ce:.3f}")

    ax1.scatter(X_test[mis_2, 0], X_test[mis_2, 1], s=100, facecolors="none", edgecolors="black", linewidths=1.5)
    ax2.scatter(X_test[mis_32, 0], X_test[mis_32, 1], s=100, facecolors="none", edgecolors="black", linewidths=1.5)
    ax2.legend(loc="best")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "failure_case_moons_under_capacity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    explanation = (
        "Failure mechanism: with only 2 hidden units, the tanh network does not have enough flexible nonlinear "
        "features to bend the decision boundary around both moon arcs. The learned separator stays too smooth and "
        "too coarse, so errors concentrate near the curved overlap region. The wider network (h=32) reshapes the "
        "feature space more effectively and reduces those boundary errors."
    )
    with open(OUTPUT_DIR / "failure_case_analysis.txt", "w", encoding="utf-8") as f:
        f.write(explanation + "\n")

    return {
        "failure_model_width": 2,
        "failure_test_acc": ev_2.test_acc,
        "failure_test_ce": ev_2.test_ce,
        "reference_test_acc": ev_32.test_acc,
        "reference_test_ce": ev_32.test_ce,
        "explanation": explanation,
    }


# =============================================================================
# Digits benchmark
# =============================================================================



def run_digits_baselines() -> dict:
    """Train the main digits baselines and save the training-dynamics figure."""
    X_tr, y_tr, X_val, y_val, X_test, y_test = load_digits_data()
    d = X_tr.shape[1]
    k = len(np.unique(y_tr))

    # Train the linear baseline first.
    softmax_model, softmax_hist = train_softmax(
        X_tr, y_tr, X_val, y_val, d=d, k=k,
        epochs=200, lr=0.05, lam=1e-4, batch_size=64, seed=42, shuffle_seed=42,
    )
    # Then train the nonlinear neural-network model on the same split.
    nn_model, nn_hist = train_nn(
        X_tr, y_tr, X_val, y_val, d=d, h=32, k=k,
        epochs=200, lam=1e-4, optimizer="adam", lr=0.001, batch_size=64,
        seed=42, shuffle_seed=42,
    )

    softmax_eval = evaluate_model(softmax_model, X_tr, y_tr, X_val, y_val, X_test, y_test, softmax_hist)
    nn_eval = evaluate_model(nn_model, X_tr, y_tr, X_val, y_val, X_test, y_test, nn_hist)

    save_training_dynamics(
        {"Softmax": softmax_hist, "NN (Adam)": nn_hist},
        OUTPUT_DIR / "digits_training_dynamics.png",
        "Digits benchmark · training dynamics",
    )

    return {
        "X_train": X_tr, "y_train": y_tr,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "d": d, "k": k,
        "softmax_model": softmax_model,
        "softmax_hist": softmax_hist,
        "softmax_eval": softmax_eval,
        "nn_model": nn_model,
        "nn_hist": nn_hist,
        "nn_eval": nn_eval,
    }



def run_digits_optimizer_study(X_tr, y_tr, X_val, y_val, X_test, y_test, d, k) -> dict:
    """Compare SGD, Momentum, and Adam for the same neural-network architecture on digits."""
    # Compare three common optimizers under the same architecture and data split.
    configs = {
        "SGD": {"optimizer": "sgd", "lr": 0.05},
        "Momentum": {"optimizer": "momentum", "lr": 0.05, "momentum": 0.9},
        "Adam": {"optimizer": "adam", "lr": 0.001},
    }
    histories = {}
    rows = []
    models = {}

    for name, cfg in configs.items():
        model, hist = train_nn(
            X_tr, y_tr, X_val, y_val, d=d, h=32, k=k,
            epochs=200, lam=1e-4, batch_size=64,
            seed=42, shuffle_seed=42, **cfg,
        )
        ev = evaluate_model(model, X_tr, y_tr, X_val, y_val, X_test, y_test, hist)
        histories[name] = hist
        models[name] = (model, hist, ev)
        rows.append({
            "optimizer": name,
            "best_epoch": ev.best_epoch,
            "val_acc": round(ev.val_acc, 4),
            "val_ce": round(ev.val_ce, 4),
            "test_acc": round(ev.test_acc, 4),
            "test_ce": round(ev.test_ce, 4),
        })

    save_training_dynamics(histories, OUTPUT_DIR / "digits_optimizer_study.png", "Digits optimizer study · NN with hidden width 32")
    write_table(OUTPUT_DIR / "digits_optimizer_study_table.txt", "Digits optimizer study", rows)
    return {"rows": rows, "models": models, "histories": histories}



def run_track_b_analysis(digits_results: dict) -> dict:
    """Generate calibration, confidence, and entropy analyses for the digits benchmark."""
    X_test = digits_results["X_test"]
    y_test = digits_results["y_test"]
    softmax_model = digits_results["softmax_model"]
    nn_model = digits_results["nn_model"]

    # Extract the test-set probabilities so we can analyze confidence and calibration.
    P_lr = softmax_model.forward(X_test)
    P_nn = nn_model.predict_proba(X_test)
    conf_lr = P_lr.max(axis=1)
    conf_nn = P_nn.max(axis=1)
    ent_lr = predictive_entropy(P_lr)
    ent_nn = predictive_entropy(P_nn)
    correct_lr = np.argmax(P_lr, axis=1) == y_test
    correct_nn = np.argmax(P_nn, axis=1) == y_test

    bc_lr, ba_lr, bn_lr, avg_conf_lr = reliability_bins(conf_lr, correct_lr)
    bc_nn, ba_nn, bn_nn, avg_conf_nn = reliability_bins(conf_nn, correct_nn)

    # Figure 1: confidence distributions

    fig2, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fig2.suptitle("Track B · confidence distribution", fontsize=13, fontweight="bold")
    for ax, conf, corr, label in [
        (axes[0], conf_lr, correct_lr, "Softmax Regression"),
        (axes[1], conf_nn, correct_nn, "Neural Network"),
    ]:
        bins = np.linspace(0.0, 1.0, 25)
        ax.hist(conf[corr], bins=bins, density=True, alpha=0.7, color=GREEN, label="Correct")
        ax.hist(conf[~corr], bins=bins, density=True, alpha=0.7, color=RED, label="Incorrect")
        ax.axvline(conf[corr].mean(), color=GREEN, ls="--", lw=1.3)
        ax.axvline(conf[~corr].mean(), color=RED, ls="--", lw=1.3)
        ax.set_title(label)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Density")
        ax.legend()
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "track_b_confidence_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # Figure 2: reliability diagrams

    fig3, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    fig3.suptitle("Track B · reliability diagrams", fontsize=13, fontweight="bold")
    for ax, centres, emp_acc, counts, label, color in [
        (axes[0], bc_lr, ba_lr, bn_lr, "Softmax Regression", BLUE),
        (axes[1], bc_nn, ba_nn, bn_nn, "Neural Network", RED),
    ]:
        ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1.3, label="Perfect calibration")
        ax.bar(centres, emp_acc, width=0.18, alpha=0.75, color=color, label="Empirical accuracy")
        for c, a, n in zip(centres, emp_acc, counts):
            ax.text(c, a + 0.025, f"n={n}", ha="center", fontsize=8, color=GRAY)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel("Confidence (bin centre)")
        ax.set_ylabel("Empirical accuracy")
        ax.set_title(label)
        ax.legend()
    fig3.tight_layout()
    fig3.savefig(OUTPUT_DIR / "track_b_reliability_diagrams.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # Figure 3: entropy comparison
    
    fig4, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fig4.suptitle("Track B · predictive entropy", fontsize=13, fontweight="bold")
    for ax, ent, corr, label in [
        (axes[0], ent_lr, correct_lr, "Softmax Regression"),
        (axes[1], ent_nn, correct_nn, "Neural Network"),
    ]:
        bins = np.linspace(0.0, max(ent.max(), 0.05), 30)
        ax.hist(ent[corr], bins=bins, density=True, alpha=0.7, color=GREEN, label="Correct")
        ax.hist(ent[~corr], bins=bins, density=True, alpha=0.7, color=RED, label="Incorrect")
        ax.axvline(ent[corr].mean(), color=GREEN, ls="--", lw=1.3)
        ax.axvline(ent[~corr].mean(), color=RED, ls="--", lw=1.3)
        ax.set_title(label)
        ax.set_xlabel("Predictive entropy")
        ax.set_ylabel("Density")
        ax.legend()
    fig4.tight_layout()
    fig4.savefig(OUTPUT_DIR / "track_b_entropy.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)

    rows = [
        {
            "model": "Softmax Regression",
            "test_acc": round(accuracy(P_lr, y_test), 4),
            "test_ce": round(cross_entropy(P_lr, y_test), 4),
            "ece": round(compute_ece(conf_lr, correct_lr, n_bins=5), 4),
            "mean_conf_correct": round(float(conf_lr[correct_lr].mean()), 4),
            "mean_conf_incorrect": round(float(conf_lr[~correct_lr].mean()), 4),
            "mean_entropy_correct": round(float(ent_lr[correct_lr].mean()), 4),
            "mean_entropy_incorrect": round(float(ent_lr[~correct_lr].mean()), 4),
        },
        {
            "model": "Neural Network",
            "test_acc": round(accuracy(P_nn, y_test), 4),
            "test_ce": round(cross_entropy(P_nn, y_test), 4),
            "ece": round(compute_ece(conf_nn, correct_nn, n_bins=5), 4),
            "mean_conf_correct": round(float(conf_nn[correct_nn].mean()), 4),
            "mean_conf_incorrect": round(float(conf_nn[~correct_nn].mean()), 4),
            "mean_entropy_correct": round(float(ent_nn[correct_nn].mean()), 4),
            "mean_entropy_incorrect": round(float(ent_nn[~correct_nn].mean()), 4),
        },
    ]
    write_table(OUTPUT_DIR / "track_b_summary_table.txt", "Track B summary", rows)

    reliability_rows = []
    for model_name, centres, emp_acc, counts, avg_conf in [
        ("Softmax Regression", bc_lr, ba_lr, bn_lr, avg_conf_lr),
        ("Neural Network", bc_nn, ba_nn, bn_nn, avg_conf_nn),
    ]:
        for c, a, n, ac in zip(centres, emp_acc, counts, avg_conf):
            reliability_rows.append({
                "model": model_name,
                "bin_center": round(float(c), 3),
                "avg_conf": round(float(ac), 4),
                "emp_acc": round(float(a), 4),
                "count": int(n),
            })
    write_table(OUTPUT_DIR / "track_b_reliability_table.txt", "Track B reliability bins", reliability_rows)

    return {
        "rows": rows,
        "reliability_rows": reliability_rows,
    }



def run_repeated_seed_digits(X_tr, y_tr, X_val, y_val, X_test, y_test, d, k) -> dict:
    """Repeat the digits benchmark across several random seeds to measure variability."""
    # Repeat the full training pipeline over multiple seeds to measure stability.
    seeds = [0, 1, 2, 3, 4]
    softmax_accs, softmax_ces = [], []
    nn_accs, nn_ces = [], []

    for s in seeds:
        m_lr, h_lr = train_softmax(
            X_tr, y_tr, X_val, y_val, d=d, k=k,
            epochs=200, lr=0.05, lam=1e-4, batch_size=64,
            seed=s, shuffle_seed=10_000 + s,
        )
        P_lr = m_lr.forward(X_test)
        softmax_accs.append(accuracy(P_lr, y_test))
        softmax_ces.append(cross_entropy(P_lr, y_test))

        m_nn, h_nn = train_nn(
            X_tr, y_tr, X_val, y_val, d=d, h=32, k=k,
            epochs=200, lam=1e-4, optimizer="adam", lr=0.001, batch_size=64,
            seed=s, shuffle_seed=20_000 + s,
        )
        P_nn = m_nn.predict_proba(X_test)
        nn_accs.append(accuracy(P_nn, y_test))
        nn_ces.append(cross_entropy(P_nn, y_test))

    softmax_accs = np.array(softmax_accs)
    softmax_ces = np.array(softmax_ces)
    nn_accs = np.array(nn_accs)
    nn_ces = np.array(nn_ces)

    rows = [
        {
            "model": "Softmax Regression",
            "mean_acc": round(float(softmax_accs.mean()), 4),
            "ci95_acc": round(t95_half_width(softmax_accs), 4),
            "mean_ce": round(float(softmax_ces.mean()), 4),
            "ci95_ce": round(t95_half_width(softmax_ces), 4),
        },
        {
            "model": "Neural Network (Adam)",
            "mean_acc": round(float(nn_accs.mean()), 4),
            "ci95_acc": round(t95_half_width(nn_accs), 4),
            "mean_ce": round(float(nn_ces.mean()), 4),
            "ci95_ce": round(t95_half_width(nn_ces), 4),
        },
    ]
    write_table(OUTPUT_DIR / "digits_repeated_seed_table.txt", "Digits repeated-seed results", rows)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    fig.suptitle("Digits · repeated-seed statistics (5 seeds)", fontsize=13, fontweight="bold")
    for ax, left, right, ylabel in [
        (axes[0], softmax_accs, nn_accs, "Test accuracy"),
        (axes[1], softmax_ces, nn_ces, "Test cross-entropy"),
    ]:
        bp = ax.boxplot([left, right], labels=["Softmax\nReg", "NN\nAdam"], patch_artist=True, widths=0.45,
                        medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor("#93C5FD")
        bp["boxes"][1].set_facecolor("#FCA5A5")
        ax.scatter(np.ones_like(left), left, c=BLUE, s=36, zorder=3)
        ax.scatter(np.ones_like(right) * 2, right, c=RED, s=36, zorder=3)
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "digits_repeated_seed_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "rows": rows,
        "softmax_accs": softmax_accs.tolist(),
        "softmax_ces": softmax_ces.tolist(),
        "nn_accs": nn_accs.tolist(),
        "nn_ces": nn_ces.tolist(),
    }


# =============================================================================
# Main
# =============================================================================



def main() -> None:
    """Run sanity checks, all experiments, and save the final summary metrics."""
    # Collect the most important metrics from every experiment in one JSON-friendly dictionary.
    summary: dict[str, object] = {}

    # -------------------------------------------------------------------------
    # Sanity checks
    # -------------------------------------------------------------------------
    X_tr_d, y_tr_d, X_val_d, y_val_d, X_test_d, y_test_d = load_digits_data()
    X_small = X_tr_d[:8]
    y_small = y_tr_d[:8]

    sm_check = SoftmaxRegression(d=X_small.shape[1], k=10, lam=1e-4, lr=0.05, batch_size=8, seed=7, shuffle_seed=7)
    nn_check = OneHiddenLayerNet(d=X_small.shape[1], h=8, k=10, lam=1e-4, optimizer="adam", lr=0.001, batch_size=8, seed=7, shuffle_seed=7)

    grad_err_softmax = finite_difference_softmax(sm_check, X_small, y_small)
    grad_err_nn = finite_difference_nn(nn_check, X_small, y_small)
    overfit_acc = overfit_small_batch(
        OneHiddenLayerNet(d=X_tr_d.shape[1], h=32, k=10, lam=1e-4, optimizer="adam", lr=0.01, batch_size=20, seed=11, shuffle_seed=11),
        X_tr_d, y_tr_d, epochs=500,
    )
    sanity = {
        "softmax_grad_relative_error": grad_err_softmax,
        "nn_grad_relative_error": grad_err_nn,
        "small_batch_overfit_acc": overfit_acc,
        "probability_sum_check_example": check_probabilities(sm_check.forward(X_small)),
        "nan_inf_check_example": not has_nan_or_inf(sm_check.forward(X_small)),
    }
    with open(OUTPUT_DIR / "sanity_checks.json", "w", encoding="utf-8") as f:
        json.dump(sanity, f, indent=2)
    summary["sanity_checks"] = sanity

    # -------------------------------------------------------------------------
    # Core experiments: linear Gaussian + moons + digits
    # -------------------------------------------------------------------------
    linear_results = run_synthetic_core_experiment("linear_gaussian", nn_width=32)
    moons_results = run_synthetic_core_experiment("moons", nn_width=32)
    summary["linear_gaussian"] = {
        "softmax_test_acc": linear_results["softmax_eval"].test_acc,
        "softmax_test_ce": linear_results["softmax_eval"].test_ce,
        "nn_test_acc": linear_results["nn_eval"].test_acc,
        "nn_test_ce": linear_results["nn_eval"].test_ce,
    }
    summary["moons"] = {
        "softmax_test_acc": moons_results["softmax_eval"].test_acc,
        "softmax_test_ce": moons_results["softmax_eval"].test_ce,
        "nn_test_acc": moons_results["nn_eval"].test_acc,
        "nn_test_ce": moons_results["nn_eval"].test_ce,
    }

    capacity_results = run_moons_capacity_ablation()
    summary["moons_capacity_ablation"] = capacity_results["rows"]

    failure_case = run_failure_case_analysis(capacity_results)
    summary["failure_case"] = failure_case

    digits_results = run_digits_baselines()
    summary["digits_baselines"] = {
        "softmax_test_acc": digits_results["softmax_eval"].test_acc,
        "softmax_test_ce": digits_results["softmax_eval"].test_ce,
        "nn_test_acc": digits_results["nn_eval"].test_acc,
        "nn_test_ce": digits_results["nn_eval"].test_ce,
    }

    optimizer_results = run_digits_optimizer_study(
        digits_results["X_train"], digits_results["y_train"],
        digits_results["X_val"], digits_results["y_val"],
        digits_results["X_test"], digits_results["y_test"],
        digits_results["d"], digits_results["k"],
    )
    summary["digits_optimizer_study"] = optimizer_results["rows"]

    track_b_results = run_track_b_analysis(digits_results)
    summary["track_b"] = track_b_results["rows"]

    repeated_seed_results = run_repeated_seed_digits(
        digits_results["X_train"], digits_results["y_train"],
        digits_results["X_val"], digits_results["y_val"],
        digits_results["X_test"], digits_results["y_test"],
        digits_results["d"], digits_results["k"],
    )
    summary["digits_repeated_seed"] = repeated_seed_results["rows"]

    with open(OUTPUT_DIR / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nComplete experiment suite finished.")
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
