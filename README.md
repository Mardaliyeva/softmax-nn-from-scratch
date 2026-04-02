
# 🧠 Simple Learning Systems: Linear vs. Nonlinear Classifiers

This project, developed as part of the **Math4AI** final capstone project, presents a geometric and statistical comparison between linear and nonlinear classifiers. The primary objective is to investigate under what conditions a one-hidden-layer neural network outperforms linear Softmax regression and to analyze the underlying mathematical reasons for this performance gap.

## 🏗 Repository Structure and Development Stages

The repository is organized into 6 main directories, each representing a specific phase of the research:

### 1. `data/` — Datasets
This folder contains the three primary datasets used in the study:
* **Linear Gaussian:** Synthetic data where classes are linearly separable.
* **Moons:** Crescent-shaped data that is impossible to separate linearly.
* **Digits:** 8x8 pixel handwritten digits (scikit-learn benchmark).

### 2. `src/` — Core Implementation
The "brain" of the project resides here. Everything was built from scratch using **NumPy**, without the use of high-level deep learning libraries (e.g., PyTorch or TensorFlow):
* **Models:** Implementation of Softmax regression and a Neural Network architecture with `tanh` activation.
* **Backpropagation:** Vectorized gradient derivations and the application of the chain rule.
* **Optimizers:** Implementation of SGD, Momentum, and Adam algorithms.

### 3. `scripts/` — Utility Tools
Scripts used to verify model correctness and implementation integrity:
* **Gradient Check:** Comparison of analytical gradients with numerical gradients (ensuring error < $10^{-9}$).
* **Sanity Checks:** Verifying the model's ability to "overfit" small datasets to ensure learning capacity.

### 4. `figures/` — Visualizations
All plots and graphs used in the report are stored here:
* **Decision Boundaries:** A comparison of the decision surfaces learned by each model.
* **Training Dynamics:** Loss and accuracy curves plotted over training epochs.

### 5. `results/` — Metrics and Analysis
Numerical results obtained after training:
* **Repeated-seed Stats:** Statistics across 5 different random seeds to ensure results are statistically significant.
* **Track B (Reliability):** Reliability diagrams showing the calibration and confidence levels of the models.

### 6. `report/` and `slides/` — Documentation
* **LaTeX Report:** The official report containing all mathematical derivations and detailed methodology.
* **Pitch Deck:** Presentation files prepared for investors or project juries.

---

## 📊 Summary of Key Results

| Task | Softmax Accuracy | Neural Network Accuracy | Conclusion |
| :--- | :---: | :---: | :--- |
| **Linear Gaussian** | 93.8% | 95.0% | Linear model is sufficient. |
| **Moons Task** | 85.0% | 95.0% | Hidden layer resolves nonlinearity. |
| **Digits (Adam)** | 94.02% | 95.65% | NN is more accurate and stable. |

## 🎓 Authors
This project was implemented by:
* Aytan
* Madina
* Ulkar

---
