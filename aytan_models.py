# aytan_models.py
# -------------------------
# SoftmaxRegression and OneHiddenLayerNet model implementations.
# Gradient checking and sanity check functions.
# Written by: Aytan
# -------------------------

from __future__ import annotations
import numpy as np
from ulker_utils import softmax, cross_entropy, accuracy, one_hot


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
        rng = np.random.default_rng(seed)
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
    ) -> dict:
        """Train for a fixed number of epochs and keep the parameters with the best validation cross-entropy."""
        rng = np.random.default_rng(self.shuffle_seed)
        n = len(y_tr)
        best_val_ce = np.inf
        best_params = (self.W.copy(), self.b.copy())
        best_epoch = 0
        history = {"train_ce": [], "val_ce": [], "train_acc": [], "val_acc": [], "best_epoch": 0}

        for ep in range(epochs):
            idx = rng.permutation(n)
            for i in range(0, n, self.bs):
                batch = idx[i:i + self.bs]
                self.step(X_tr[batch], y_tr[batch])

            P_tr = self.forward(X_tr)
            P_val = self.forward(X_val)
            tr_ce = cross_entropy(P_tr, y_tr)
            val_ce = cross_entropy(P_val, y_val)

            history["train_ce"].append(tr_ce)
            history["val_ce"].append(val_ce)
            history["train_acc"].append(accuracy(P_tr, y_tr))
            history["val_acc"].append(accuracy(P_val, y_val))

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
        rng = np.random.default_rng(seed)
        self.h = h
        self.k = k
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

    def forward(self, X: np.ndarray):
        """Run the full forward pass and return intermediate activations for backpropagation."""
        Z1 = X @ self.W1.T + self.b1
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
        dS = (P - Y) / n
        dW2 = dS.T @ H + self.lam * self.W2
        db2 = dS.sum(axis=0)
        dH = dS @ self.W2
        dZ1 = dH * (1.0 - H ** 2)
        dW1 = dZ1.T @ X + self.lam * self.W1
        db1 = dZ1.sum(axis=0)
        return dW1, db1, dW2, db2

    def _update_parameter(self, param, grad, v_buf, m_buf, s_buf):
        """Update one parameter tensor using the selected optimizer rule."""
        if self.optimizer == "sgd":
            param = param - self.lr * grad
            return param, v_buf, m_buf, s_buf
        if self.optimizer == "momentum":
            v_new = self.momentum_coef * v_buf + self.lr * grad
            param = param - v_new
            return param, v_new, m_buf, s_buf
        if self.optimizer == "adam":
            m_new = self.beta1 * m_buf + (1.0 - self.beta1) * grad
            s_new = self.beta2 * s_buf + (1.0 - self.beta2) * (grad ** 2)
            m_hat = m_new / (1.0 - self.beta1 ** self.t)
            s_hat = s_new / (1.0 - self.beta2 ** self.t)
            param = param - self.lr * m_hat / (np.sqrt(s_hat) + self.eps_adam)
            return param, v_buf, m_new, s_new
        raise ValueError(f"Unknown optimizer: {self.optimizer}")

    def step(self, X: np.ndarray, y: np.ndarray) -> None:
        """Take one optimization step on a mini-batch."""
        self.t += 1
        dW1, db1, dW2, db2 = self.gradients(X, y)
        self.W1, self.vW1, self.mW1, self.sW1 = self._update_parameter(self.W1, dW1, self.vW1, self.mW1, self.sW1)
        self.b1, self.vb1, self.mb1, self.sb1 = self._update_parameter(self.b1, db1, self.vb1, self.mb1, self.sb1)
        self.W2, self.vW2, self.mW2, self.sW2 = self._update_parameter(self.W2, dW2, self.vW2, self.mW2, self.sW2)
        self.b2, self.vb2, self.mb2, self.sb2 = self._update_parameter(self.b2, db2, self.vb2, self.mb2, self.sb2)

    def train(self, X_tr, y_tr, X_val, y_val, epochs: int = 200) -> dict:
        """Train the neural network and restore the checkpoint with the best validation cross-entropy."""
        rng = np.random.default_rng(self.shuffle_seed)
        n = len(y_tr)
        best_val_ce = np.inf
        best_epoch = 0
        best_params = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())
        history = {"train_ce": [], "val_ce": [], "train_acc": [], "val_acc": [], "best_epoch": 0}

        for ep in range(epochs):
            idx = rng.permutation(n)
            for i in range(0, n, self.bs):
                batch = idx[i:i + self.bs]
                self.step(X_tr[batch], y_tr[batch])

            P_tr = self.predict_proba(X_tr)
            P_val = self.predict_proba(X_val)
            tr_ce = cross_entropy(P_tr, y_tr)
            val_ce = cross_entropy(P_val, y_val)

            history["train_ce"].append(tr_ce)
            history["val_ce"].append(val_ce)
            history["train_acc"].append(accuracy(P_tr, y_tr))
            history["val_acc"].append(accuracy(P_val, y_val))

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

    def check_param(array, grad):
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
