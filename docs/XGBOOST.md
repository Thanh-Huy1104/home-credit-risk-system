# XGBoost - Complete Mathematical Guide

## Table of Contents
1. [What is XGBoost?](#1-what-is-xgboost)
2. [Decision Trees Refresher](#2-decision-trees-refresher)
3. [Gradient Boosting Intuition](#3-gradient-boosting-intuition)
4. [XGBoost Mathematical Formulation](#4-xgboost-mathematical-formulation)
5. [Loss Functions](#5-loss-functions)
6. [Regularization](#6-regularization)
7. [Split Finding](#7-split-finding)
8. [Leaf Weights](#8-leaf-weights)
9. [Feature Importance](#9-feature-importance)
10. [Key Hyperparameters](#10-key-hyperparameters)
11. [Categorical Features](#11-categorical-features)
12. [Practical Workflow](#12-practical-workflow)

---

## 1. What is XGBoost?

**XGBoost (eXtreme Gradient Boosting)** is an optimized distributed gradient boosting library.

### Key Characteristics:
- **Ensemble method**: Combines many weak learners (trees)
- **Boosting**: Builds trees sequentially, each correcting the previous
- **Gradient-based**: Uses gradients to find optimal splits
- **Regularized**: Built-in L1/L2 regularization
- **Efficient**: Uses histogram-based splitting and parallelization

---

## 2. Decision Trees Refresher

### What is a Decision Tree?

A tree that splits data based on feature values to make predictions.

```
                    age > 30?
                   /         \
                 Yes          No
                 ↓            ↓
            income > 50K?   → Low Risk
            /        \
          Yes         No
           ↓          ↓
       High Risk   Medium Risk
```

### Key Terminology:

| Term | Meaning |
|------|---------|
| **Root** | Top node (first split) |
| **Leaf** | Terminal node (prediction) |
| **Depth** | Distance from root |
| **Split** | Decision rule at a node |
| **Branch** | Path from root to leaf |

### Prediction at Leaf

For a classification problem, each leaf predicts a **class probability** or **class label**.

For regression, each leaf predicts a **continuous value**.

---

## 3. Gradient Boosting Intuition

### The Core Idea

Instead of building one big tree, build many small trees sequentially:

1. **Tree 1**: Make predictions, some are wrong
2. **Tree 2**: Focus on fixing Tree 1's mistakes
3. **Tree 3**: Fix what Tree 2 got wrong
4. ... continue for n trees

**Final prediction = sum of all tree predictions**

### Visual Intuition

```
Prediction = Tree1 + Tree2 + Tree3 + ... + TreeN
            ↓           ↓        ↓
         0.3         +0.1     +0.05   = 0.45
```

### What are we "boosting"?

We compute **residuals** (or **gradients**) - how wrong our predictions are:

```
residual_i = y_true_i - prediction_i
```

Each new tree tries to predict these residuals!

---

## 4. XGBoost Mathematical Formulation

### The Objective Function

XGBoost optimizes:

```
Obj = Loss + Regularization

Obj = Σ L(y_i, ŷ_i) + Σ Ω(f_k)

Where:
- L = loss function
- ŷ_i = prediction for sample i
- f_k = k-th tree
- Ω = regularization on tree structure
```

### Additive Training

After building k-1 trees, prediction is:
```
ŷ_i^(k-1) = Σ_{t=1}^{k-1} f_t(x_i)
```

We want to add tree f_k:
```
ŷ_i^(k) = ŷ_i^(k-1) + f_k(x_i)
```

### The Magic: Taylor Expansion

To find the best f_k, we use **second-order Taylor expansion**:

```
L(y_i, ŷ_i^(k)) ≈ L(y_i, ŷ_i^(k-1)) + g_i * f_k(x_i) + 0.5 * h_i * f_k(x_i)²

Where:
- g_i = first derivative (gradient) of loss w.r.t. prediction
- h_i = second derivative (hessian) of loss w.r.t. prediction

g_i = ∂L/∂ŷ    (gradient)
h_i = ∂²L/∂ŷ²  (hessian)
```

### For Log-Loss (Binary Classification)

```
L = -[y * log(p) + (1-y) * log(1-p)]

Where p = sigmoid(ŷ)

Gradient (g_i):
g_i = p_i - y_i

Hessian (h_i):  
h_i = p_i * (1 - p_i)
```

**Example:**
```
y = 1 (actual default)
p = 0.3 (predicted probability)

g = 0.3 - 1 = -0.7  (negative = under-predicting)
h = 0.3 * 0.7 = 0.21
```

---

## 5. Loss Functions

### For Different Problems

#### Binary Classification
```
L = -[y * log(p) + (1-y) * log(1-p)]

g = p - y
h = p * (1 - p)
```

#### Multi-Class Classification
```
L = -Σ y_c * log(p_c)

Uses softmax for p_c
```

#### Regression (Squared Error)
```
L = 0.5 * (y - ŷ)²

g = ŷ - y
h = 1 (constant!)
```

#### Regression (Huber) - Less sensitive to outliers
```
L = {
    0.5 * (y - ŷ)²              if |y - ŷ| ≤ δ
    δ * |y - ŵ| - 0.5 * δ²      otherwise
}
```

---

## 6. Regularization

### Why Regularize Trees?

Without regularization, trees can perfectly fit training data (overfitting).

### XGBoost's Tree Regularization

```
Ω(f) = γ * T + 0.5 * λ * Σ w_j²

Where:
- T = number of leaves in tree
- w_j = weight (prediction) of leaf j
- γ = minimum loss reduction to make a split (complexity penalty)
- λ = L2 regularization on leaf weights
```

### Effect of Regularization

| Parameter | Effect |
|-----------|--------|
| **γ (gamma)** | Higher → fewer splits → simpler trees |
| **λ (lambda)** | Higher → smaller leaf weights → less overfitting |
| **α (alpha)** | L1 regularization → sparse solutions |

### Example

```
Without regularization:  leaf weight = 2.5
With λ=1:              leaf weight = 2.5 / (1 + 1) = 1.25

Formula: w* = -G / (H + λ)

Where G = sum of gradients, H = sum of hessians
```

---

## 7. Split Finding

### How XGBoost Finds the Best Split

For each feature, XGBoost evaluates all possible split points:

```
For feature x at split s:
  
  Left:  {i | x_i ≤ s}
  Right: {i | x_i > s}
  
  G_L = Σ g_i for i in left
  H_L = Σ h_i for left
  
  G_R = Σ g_i for i in right  
  H_R = Σ h_i for right
  
  Gain = 0.5 * [G_L²/(H_L + λ) + G_R²/(H_R + λ) - (G_L+G_R)²/(H_L+H_R+λ)] - γ
```

### The Gain Formula

```
Gain = (gain with split) - (gain without split) - γ

Higher gain = better split = reduces loss more
```

### Histogram-Based Splitting (Optimization)

Instead of trying all split points:

1. Bin continuous features into discrete bins (e.g., 256 bins)
2. Only evaluate splits at bin boundaries
3. Much faster with minimal accuracy loss

---

## 8. Leaf Weights

### Computing Leaf Weight

Once we've found the best split structure, compute weight for each leaf:

```
w_j* = -Σ g_i / (Σ h_i + λ)

For each leaf j:
  - Sum gradients (g_i) of samples in that leaf
  - Sum hessians (h_i) of samples in that leaf
  - Apply L2 regularization (λ)
```

### For Binary Classification (Special Case)

```
w_j* = (count_class_1 - count_class_0) / (count_total + λ)

Simplified: average of residuals in leaf, regularized
```

### Prediction Update

```
Final prediction for sample i:
ŷ_i = initial_prediction + η * Σ w_leaf(x_i)

Where η = learning_rate (shrinkage factor)
```

---

## 9. Feature Importance

### Types of Importance

#### 1. Weight (Cover)
```
Importance = number of times feature is used in splits
```

#### 2. Gain
```
Importance = sum of gain from all splits using this feature

Gain = average loss reduction from splits using this feature
```

#### 3. Total Gain
```
Importance = Σ gain_i for all splits
```

#### 4. SHAP Values (Most Accurate)
```
Accounts for interaction effects
Complex but most theoretically sound
```

---

## 10. Key Hyperparameters

### Tree Structure

| Parameter | Range | Effect |
|-----------|-------|--------|
| `max_depth` | 1-20 | Max tree depth. Higher = more complex |
| `min_child_weight` | 0-∞ | Min sum of hessian in leaf |
| `gamma` | 0-∞ | Min loss reduction for split |
| `max_bin` | 1-256 | Number of bins for histogram |

### Learning

| Parameter | Range | Effect |
|-----------|-------|--------|
| `learning_rate` (η) | 0-1 | Shrinkage per tree. Lower = more trees |
| `n_estimators` | 1-∞ | Number of trees |
| `early_stopping_rounds` | 1-∞ | Stop if no improvement |

### Regularization

| Parameter | Range | Effect |
|-----------|-------|--------|
| `lambda` (L2) | 0-∞ | L2 regularization |
| `alpha` (L1) | 0-∞ | L1 regularization |
| `subsample` | 0-1 | Row sampling per tree |
| `colsample_bytree` | 0-1 | Column sampling per tree |

### Practical Defaults

```python
model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0,
    reg_lambda=1,
    early_stopping_rounds=50
)
```

---

## 11. Categorical Features

### Traditional: One-Hot Encoding

```
Category: ["Cat", "Dog", "Bird"]
        ↓
Cat: [1, 0, 0]
Dog: [0, 1, 0]
Bird: [0, 0, 1]

Problems:
- Sparse matrices
- No ordinal relationship captured
- Memory expensive
```

### XGBoost Native: Optimal Partitioning

1. **Sort categories** by target mean
2. **Find optimal split** points between categories
3. **Learn category ordering** from data

### Mathematical Treatment

```
For category c with n_c samples:
  gradient_c = Σ g_i / n_c
  
Sort categories by gradient_c, then find splits
that maximize gain between adjacent categories
```

### Usage

```python
# Convert to category dtype
df['column'] = df['column'].astype('category')

# Train with native support
model = xgb.XGBClassifier(enable_categorical=True)
```

---

## 12. Practical Workflow

### Step-by-Step

```
1. Prepare data
   ├── Handle missing values (XGBoost handles natively)
   ├── Convert categoricals to 'category' dtype
   └── Split into train/validation

2. Calculate class weight (if imbalanced)
   └── scale_pos_weight = neg_count / pos_count

3. Initialize with baseline
   └── Initial prediction = log(neg/pos)  (for classification)

4. Build trees iteratively
   ├── Compute gradients
   ├── Find best splits (gain)
   ├── Compute leaf weights
   ├── Update predictions with shrinkage

5. Early stopping
   └── Stop if validation loss doesn't improve

6. Evaluate
   ├── ROC-AUC for ranking
   ├── Precision/Recall for thresholds
   └── Feature importance analysis
```

### Pseudocode

```python
# Simplified XGBoost training loop
for tree in range(n_estimators):
    
    # 1. Compute pseudo-residuals (gradients)
    predictions = current_predictions
    gradients = compute_gradient(y_true, predictions)
    hessians = compute_hessian(y_true, predictions)
    
    # 2. Find best split for each node
    for each_feature:
        for each_split_point:
            gain = compute_gain(left_grads, right_grads, 
                               left_hess, right_hess, lambda)
            if gain > best_gain:
                best_split = this_split
    
    # 3. Create tree structure
    tree = build_tree(best_splits)
    
    # 4. Compute leaf weights
    for each_leaf:
        weight = -sum(gradients) / (sum(hessians) + lambda)
    
    # 5. Update predictions
    predictions += learning_rate * weight
    
    # 6. Check early stopping
    if validation_loss > best_loss for rounds:
        break
```

---

## Summary Cheat Sheet

| Concept | Formula | Meaning |
|---------|---------|---------|
| **Gradient** | g = ∂L/∂ŷ | Direction to reduce loss |
| **Hessian** | h = ∂²L/∂ŷ² | Curvature of loss |
| **Gain** | Δ = G_L²/(H_L+λ) + G_R²/(H_R+λ) - G²/(H+λ) | Improvement from split |
| **Leaf Weight** | w = -Σg / (Σh + λ) | Prediction at leaf |
| **Prediction** | ŷ = ŷ₀ + ηΣw | Final output |
| **Regularization** | Ω = γT + ½λΣw² | Penalty on complexity |

---

## Quick Reference: Binary Classification

```
Loss: L = -[y log(p) + (1-y) log(1-p)]

p = 1 / (1 + e^(-ŷ))

Gradient: g = p - y
Hessian:  h = p(1-p)

Leaf weight: w = -Σ(p-y) / Σ(p(1-p) + λ)

Positive weight: scale_pos_weight = neg/pos
```

This ensures higher weight for minority class!
