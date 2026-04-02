# -------------------------
# Experiment pipeline: synthetic tasks, digits benchmark, ablations,
# optimizer study, Track B analysis, repeated-seed evaluation, and main().
# Written by: Medine
# -------------------------

from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from ulker_utils import (
    OUTPUT_DIR,
    load_digits_data, load_synthetic_dataset,
    accuracy, cross_entropy, predictive_entropy,
    compute_ece, reliability_bins, t95_half_width,
    check_probabilities, has_nan_or_inf,
    evaluate_model, train_softmax, train_nn, write_table,
)
from aytan_models import (
    SoftmaxRegression, OneHiddenLayerNet,
    finite_difference_softmax, finite_difference_nn, overfit_small_batch,
)

# =============================================================================
# Plot colors and settings
# =============================================================================

BLUE   = "#2563EB"
RED    = "#DC2626"
GREEN  = "#16A34A"
GRAY   = "#6B7280"
AMBER  = "#D97706"
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


# =============================================================================
# Plotting helpers
# =============================================================================


def _predict_proba(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X) if hasattr(model, "predict_proba") else model.forward(X)


def plot_binary_boundary(ax, model, X, y, title: str, misclassified=None) -> None:
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    P = _predict_proba(model, grid)
    Z = P[:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), cmap="RdBu", alpha=0.32)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1.2)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=22, c=BLUE, edgecolor="white", linewidth=0.5, label="Class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=22, c=RED, edgecolor="white", linewidth=0.5, label="Class 1")
    if misclassified is not None and np.any(misclassified):
        ax.scatter(X[misclassified, 0], X[misclassified, 1],
                   s=90, facecolors="none", edgecolors="black", linewidths=1.4, label="Misclassified")
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def save_training_dynamics(histories: dict, outpath: Path, title: str) -> None:
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
# Synthetic experiments
# =============================================================================


def run_synthetic_core_experiment(dataset_name: str, nn_width: int = 32) -> dict:
    """Train both models on one synthetic task, evaluate them, and save a decision-boundary comparison."""
    X_tr, y_tr, X_val, y_val, X_test, y_test = load_synthetic_dataset(dataset_name)
    d = X_tr.shape[1]
    k = len(np.unique(y_tr))

    softmax_model, softmax_hist = train_softmax(
        X_tr, y_tr, X_val, y_val, d=d, k=k,
        epochs=300, lr=0.05, lam=1e-4, batch_size=32, seed=42, shuffle_seed=42,
    )
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
            "width": h, "best_epoch": ev.best_epoch,
            "val_acc": round(ev.val_acc, 4), "val_ce": round(ev.val_ce, 4),
            "test_acc": round(ev.test_acc, 4), "test_ce": round(ev.test_ce, 4),
        })

    axes[-1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "moons_capacity_ablation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    write_table(OUTPUT_DIR / "moons_capacity_ablation_table.txt", "Moons capacity ablation", results)
    return {"rows": results, "trained_models": trained_models, "all_X": all_X, "all_y": all_y, "X_test": X_test, "y_test": y_test}


def run_failure_case_analysis(moons_ablation: dict) -> dict:
    """Visualize and explain how an under-capacity network fails on the moons dataset."""
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

    softmax_model, softmax_hist = train_softmax(
        X_tr, y_tr, X_val, y_val, d=d, k=k,
        epochs=200, lr=0.05, lam=1e-4, batch_size=64, seed=42, shuffle_seed=42,
    )
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
        "softmax_model": softmax_model, "softmax_hist": softmax_hist, "softmax_eval": softmax_eval,
        "nn_model": nn_model, "nn_hist": nn_hist, "nn_eval": nn_eval,
    }


def run_digits_optimizer_study(X_tr, y_tr, X_val, y_val, X_test, y_test, d, k) -> dict:
    """Compare SGD, Momentum, and Adam for the same neural-network architecture on digits."""
    configs = {
        "SGD": {"optimizer": "sgd", "lr": 0.05},
        "Momentum": {"optimizer": "momentum", "lr": 0.05, "momentum": 0.9},
        "Adam": {"optimizer": "adam", "lr": 0.001},
    }
    histories, rows, models = {}, [], {}

    for name, cfg in configs.items():
        model, hist = train_nn(
            X_tr, y_tr, X_val, y_val, d=d, h=32, k=k,
            epochs=200, lam=1e-4, batch_size=64, seed=42, shuffle_seed=42, **cfg,
        )
        ev = evaluate_model(model, X_tr, y_tr, X_val, y_val, X_test, y_test, hist)
        histories[name] = hist
        models[name] = (model, hist, ev)
        rows.append({
            "optimizer": name, "best_epoch": ev.best_epoch,
            "val_acc": round(ev.val_acc, 4), "val_ce": round(ev.val_ce, 4),
            "test_acc": round(ev.test_acc, 4), "test_ce": round(ev.test_ce, 4),
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

    return {"rows": rows, "reliability_rows": reliability_rows}


def run_repeated_seed_digits(X_tr, y_tr, X_val, y_val, X_test, y_test, d, k) -> dict:
    """Repeat the digits benchmark across several random seeds to measure variability."""
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
    summary: dict = {}

    # Sanity checks
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

    # Core experiments
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
