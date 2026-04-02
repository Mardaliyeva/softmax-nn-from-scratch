
# 📘 Math4AI Final Capstone Project
---

# 📝 Project Overview

This project studies a fundamental question in machine learning:

*When does a nonlinear neural network actually outperform a simple linear classifier?*

To answer this question, we compare two models:

1. *Multiclass Softmax Regression (Linear Model)*
2. *One-Hidden-Layer Neural Network with tanh activation*

Both models are implemented *from scratch using NumPy*, and evaluated on three datasets with different geometric structures.

The goal is not simply to achieve high accuracy, but to understand the *mathematical reasons why nonlinear models can sometimes outperform linear models.*

---

# 📐 Mathematical Background

This project is deeply connected to concepts from:

### 🟦 Linear Algebra

Used to represent data and model parameters.

* Input data is represented as vectors
* Datasets are matrices
* Model computations use matrix multiplication

Example representation:

X ∈ ℝ^(n × d)

where

* *n* = number of samples
* *d* = number of features

Model parameters are also matrices:

W ∈ ℝ^(k × d)

These matrices transform input vectors into prediction scores.

---

### 🎲 Probability Theory

Classification models output *probability distributions* over classes.

The *softmax function* converts linear scores into probabilities:

[
p_j(x) = \frac{e^{s_j(x)}}{\sum_\ell e^{s_\ell(x)}}
]

Properties:

* All probabilities ≥ 0
* Probabilities sum to 1

The predicted class is the class with *highest probability*.

Prediction confidence is defined as:

confidence(x) = max_j p_j(x)

This concept is used in our *Track B analysis of prediction confidence and reliability.*

---

### 🔢 Calculus (Optimization)

The model learns by *minimizing cross-entropy loss*.

For a correct class (y):

[
L(x,y) = -\log p_y(x)
]

This loss penalizes predictions that assign low probability to the correct class.

Training uses *gradient descent*, which requires derivatives:

* Compute gradient of loss with respect to parameters
* Update parameters in direction that decreases loss

Backpropagation applies the *chain rule* repeatedly through the network.

---

### 📊 Statistics

Statistics is used in several important parts:

#### Dataset Splitting

The dataset is divided into:

* *Training set* – learn model parameters
* *Validation set* – choose hyperparameters
* *Test set* – final evaluation

This ensures an *unbiased estimate of model performance*.

#### Evaluation Metrics

Model performance is measured using:

* Accuracy
* Cross-Entropy
* Precision
* Recall
* F1 score
* ROC curve
* Calibration metrics

These statistical metrics help measure *prediction correctness and reliability*.

---

# 🖥️ Models Implemented

## 1. Softmax Regression (Linear Classifier)

The model computes:

[
s(x) = Wx + b
]

where:

* (W) = weight matrix
* (b) = bias vector

Softmax converts scores to probabilities.

Decision boundaries are *linear hyperplanes*.

This model works well when the true class boundary is approximately linear.

---

## 2. One-Hidden-Layer Neural Network

Architecture:

Input → Hidden Layer (tanh) → Output → Softmax

Mathematically:

[
h = tanh(W_1 x + b_1)
]

[
s = W_2 h + b_2
]

The hidden layer creates a *nonlinear feature transformation, allowing the model to represent **curved decision boundaries*.

---

# 🗂️ Datasets

Three datasets were used to study different geometric structures.

## 1. Linear Gaussian Dataset

Two Gaussian clusters with mild overlap.

Expected behavior:

* Linear decision boundary
* Softmax regression should perform well

Result:

Both models achieve similar accuracy (~94%).

This confirms that *additional model complexity is unnecessary when the geometry is linear*.

---

## 2. Moons Dataset

Two interleaving crescent shapes.

This dataset *cannot be separated by a linear boundary*.

Result:

| Model          | Test Accuracy |
| -------------- | ------------- |
| Softmax        | 0.85          |
| Neural Network | 0.95          |

The neural network learns a *curved boundary*, demonstrating the advantage of nonlinear feature transformation.

---

## 3. Digits Dataset

Handwritten digit images.

Properties:

* 1797 samples
* 64 features (8×8 pixels)
* 10 classes

Results:

| Model              | Test Accuracy |
| ------------------ | ------------- |
| Softmax Regression | 94.02%        |
| Neural Network     | 95.65%        |

The neural network also achieves *lower cross-entropy*, meaning it produces more confident predictions.

---

# 🧪 Experiments

## Capacity Ablation

Hidden layer widths tested:

2
8
32

Results show that:

* width 2 cannot represent complex boundaries
* width 8 performs well
* width 32 offers minimal additional improvement

This demonstrates that *model capacity must match dataset complexity*.

---

## Optimizer Study

Three optimizers were tested:

* SGD
* Momentum
* Adam

Results:

Adam converges faster and achieves the lowest cross-entropy.

---

## Failure Case Analysis

A neural network with *hidden width = 2* fails on the moons dataset.

Reason:

Two hidden units cannot produce enough nonlinear features to represent the curved boundary.

This demonstrates a *representational limitation*, not an optimization failure.

---

# 🔒 Reliability and Confidence Analysis (Track B)

The project also evaluates *prediction reliability*.

Metrics include:

### Prediction Confidence

confidence(x) = max probability

### Predictive Entropy

Measures uncertainty of probability distribution.

Low entropy → confident prediction
High entropy → uncertain prediction

### Expected Calibration Error (ECE)

Measures how well predicted confidence matches actual accuracy.

Results:

| Model          | ECE   |
| -------------- | ----- |
| Softmax        | 0.097 |
| Neural Network | 0.012 |

The neural network is *better calibrated*.

---

# 📂 Repository Structure

Project_AIAcademy/
│
├── aytan_models.py
│   Model implementations
│
├── ulker_utils.py
│   Shared utilities
│   - softmax
│   - cross entropy
│   - accuracy
│   - calibration helpers
│
├── medine_experiments.py
│   Experimental pipeline
│
├── project_son_code.py
│   Main execution script
│
├── data/
│   datasets (.npz files)
│
├── capstone_outputs/
│   generated experiment results
│
├── project_report.docx
│
└── project_presentation.pptx

---

# 🤝 Team Contributions

### Aytan

Responsible for *model implementation*.

Implemented:

* SoftmaxRegression classifier
* OneHiddenLayerNet neural network
* Forward propagation
* Backpropagation
* Optimizers (SGD, Momentum, Adam)
* Gradient checking

---

### Ulkar

Responsible for *data pipeline and shared utilities*.

Implemented:

* dataset loading
* train/validation/test split reconstruction
* softmax function
* cross-entropy loss
* accuracy computation
* evaluation metrics
* calibration and uncertainty helpers
* reusable training wrappers

---

### Madina

Responsible for *experimental pipeline*.

Implemented:

* core experiments
* capacity ablation
* optimizer comparison
* failure case analysis
* repeated-seed robustness evaluation
* confidence and calibration analysis

---

# 🏆 Key Findings

1. Linear classifiers perform well when the class boundary is linear.
2. Neural networks outperform linear models on nonlinear datasets.
3. Model capacity must match dataset complexity.
4. Neural networks produce better calibrated probabilities.

---

---
