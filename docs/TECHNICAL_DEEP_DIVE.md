# XGBoost Categorical Handling & Aggregations - Technical Deep Dive

## 1. How XGBoost Handles Categorical Data

### The Two Approaches

#### A. Traditional: One-Hot Encoding (Not Used Here)

Traditional approach converts categories to binary columns:
```
NAME_CONTRACT_TYPE: ["Cash loans", "Revolving loans"]
                    ↓
Cash loans: [1, 0]
Revolving loans: [0, 1]
```

**Problems:**
- Creates sparse matrices (mostly zeros)
- Assumes independence between categories (no ordinal relationship)
- Expands feature space significantly

#### B. Native Categorical Handling (What We Use)

XGBoost 1.5+ has built-in categorical support via `enable_categorical=True`:

```python
model = xgb.XGBClassifier(enable_categorical=True, ...)
```

**How it works:**

1. **Category Encoding:** XGBoost uses a technique called **optimal partition-based encoding** (similar to LightGBM's approach):

   For a categorical feature with k categories, XGBoost:
   - Sorts all categories by their target mean (for classification)
   - Finds optimal split points that maximize information gain
   - Assigns category-specific gradients

2. **Mathematical Formulation:**

   For each category c in feature x:
   
   ```
   gradient[c] = (sum of target values in category c) / (count in category c) - global_mean
   
   This is essentially: E[Y|X=c] - E[Y]
   ```

   The split at category c is evaluated using:
   
   ```
   Gain = (∑(gradient_left)² / n_left + ∑(gradient_right)² / n_right) - ∑(gradient_parent)² / n_parent
   ```

3. **Advantages:**
   - Preserves ordinal information (if meaningful)
   - No memory explosion
   - Learns optimal category ordering from data
   - Handles categories with few samples better

### Our Implementation

```python
# In train.py
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category')
```

This tells Pandas to store the column as categorical dtype, which XGBoost automatically detects and handles with its native algorithm.

---

## 2. Aggregation Strategy: Why Aggregate Before Join?

### The Problem Illustrated

**One-to-Many Relationship:**

```
Master table (1)          Supplementary table (many)
─────────────────         ──────────────────────────
SK_ID_CURR               SK_ID_CURR  Loan 1
100001                   100001      Loan 2
100002                   100001      Loan 3
                         100002      Loan 1
                         100002      Loan 2
```

**If we JOIN without aggregating:**
```sql
SELECT m.*, s.loan_amount
FROM master m
LEFT JOIN supplementary s ON m.SK_ID_CURR = s.SK_ID_CURR
```

**Result (BROKEN):**
```
SK_ID_CURR  loan_amount
100001      5000     ← Row 1
100001      7000     ← Row 2 (DUPLICATE!)
100001      3000     ← Row 3 (DUPLICATE!)
100002      4000
100002      6000
```

Now applicant 100001 appears 3 times → model learns wrong patterns!

**If we AGGREGATE first:**
```sql
SELECT SK_ID_CURR, 
       COUNT(*) as loan_count,
       SUM(loan_amount) as total_debt,
       AVG(loan_amount) as avg_loan
FROM supplementary
GROUP BY SK_ID_CURR
```

**Result (CORRECT):**
```
SK_ID_CURR  loan_count  total_debt  avg_loan
100001      3           15000       5000
100002      2           10000       5000
```

**Then JOIN:** One row per applicant!

---

## 3. Mathematical Explanation of Aggregations

### Bureau (External Credit History)

**What it represents:** Client's credit history with OTHER financial institutions (reported to Credit Bureau)

**Aggregations chosen:**

| Feature | Formula | Why |
|---------|---------|-----|
| `bureau_loan_count` | COUNT(SK_ID_BUREAU) | Total credit relationships |
| `bureau_active_count` | SUM(CASE WHEN CREDIT_ACTIVE='Active' THEN 1 END) | Currently active obligations |
| `bureau_total_debt` | SUM(AMT_CREDIT_SUM_DEBT) | Total outstanding debt |
| `bureau_active_pct` | active_count / total_count | Ratio of active to closed |
| `bureau_max_days_overdue` | MAX(CREDIT_DAY_OVERDUE) | Worst payment behavior |

**Mathematical interpretation:**

For a client i with n bureau records:
```
bureau_loan_count_i = n

bureau_total_debt_i = Σ_j debt_ij  for j in 1...n

bureau_active_pct_i = (Σ_j active_ij) / n  where active_ij ∈ {0,1}
```

These become **single scalar features** that XGBoost can use in splits.

### Bureau Balance (Monthly Status)

**What it represents:** Month-by-month status of each bureau loan

**Aggregations chosen:**

| Feature | Formula | Why |
|---------|---------|-----|
| `bb_count` | COUNT(*) | Total monthly observations |
| `bb_status_1`, `bb_status_2` | SUM(CASE WHEN STATUS='1'/'2') | Months with 1-30 / 30-60 days overdue |
| `bb_late_pct` | (status_1 + status_2) / count | Proportion of problematic months |

**Why aggregate this way:**

STATUS codes:
- '0': 1-30 days overdue
- '1': 31-60 days overdue  
- '2': 61-90 days overdue
- 'C': Closed
- 'X': No status (good)

The count of '1's and '2's directly measures **chronic payment problems**.

### Previous Application (Internal)

**What it represents:** Previous loan applications to Home Credit

**Aggregations chosen:**

| Feature | Formula | Why |
|---------|---------|-----|
| `prev_app_count` | COUNT(SK_ID_PREV) | Number of previous applications |
| `prev_app_refused` | SUM(CASE WHEN STATUS='Refused') | Count of rejections |
| `prev_app_refused_pct` | refused / count | Rejection rate |
| `prev_app_approved_pct` | approved / count | Approval rate |
| `prev_app_amount_mean` | AVG(AMT_APPLICATION) | Average requested amount |

**Why `prev_app_refused` is crucial:**

If a client was refused multiple times, they're higher risk:
- Either the algorithm found issues in past applications
- Or they desperately needed credit (red flag)

Mathematically:
```
refused_pct_i = (Σ_j I(status_ij = 'Refused')) / n_ij

where I(condition) = 1 if true, 0 otherwise
```

### Installments Payments (Repayment Behavior)

**What it represents:** Actual payment history for previous loans

**Key derived features:**

```python
PAYMENT_DIFF = AMT_INSTALMENT - AMT_PAYMENT
DAYS_LATE = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT
PAID_LATE = I(DAYS_LATE > 0)
```

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `payment_diff_mean` | AVG(AMT_INSTALMENT - AMT_PAYMENT) | Underpayment tendency |
| `days_late_mean` | AVG(DAYS_LATE) | Average days past due |
| `paid_late_pct` | SUM(PAID_LATE) / count | Frequency of late payment |

**Why these matter:**

- `payment_diff_mean < 0`: Underpaying consistently (cash flow issues)
- `payment_diff_mean > 0`: Overpaying (responsible or trying to pay down principal)
- `paid_late_pct`: Measures payment discipline

**Mathematical note:**
```
payment_diff_mean_i = (1/n) * Σ_j (inst_j - paid_j)
days_late_mean_i = (1/n) * Σ_j (entry_j - inst_j)
```

### POS Cash & Credit Card

**Similar pattern:** Count of records, status breakdown, utilization metrics

| Feature | Formula | Why |
|---------|---------|-----|
| `pos_count` | COUNT(*) | Total POS/CC records |
| `cc_balance_mean` | AVG(AMT_BALANCE) | Average balance |
| `cc_credit_limit_mean` | AVG(AMT_CREDIT_LIMIT_ACTUAL) | Credit limit |
| `cc_utilization` | balance / limit (derived) | Credit usage |

---

## 4. Null/Missing Value Handling

### Sources of Nulls

1. **Original data missing:** Client didn't provide info
2. **Not applicable:** Field doesn't apply (e.g., no previous application)
3. **Aggregation edge cases:** No records to aggregate

### XGBoost's Native Handling

XGBoost handles missing values internally:

```python
# When XGBoost encounters a missing value at node m:
# 1. It computes gain_left using only non-missing values going left
# 2. It computes gain_right using only non-missing values going right  
# 3. It compares: gain with missing in left vs. gain with missing in right
# 4. It sends missing values down the branch with higher gain
```

**Mathematical treatment:**

For a split on feature x at threshold t:
```
gain = H(parent) - (n_left/n) * H(left) - (n_right/n) * H(right)

Where H is the impurity (Gini or entropy)
Missing values don't contribute to n_left or n_right
```

**Default behavior in XGBoost:**
- Missing values go to whichever child minimizes loss
- Can also use `missing` parameter to specify default direction

### Our COALESCE Strategy

In the join query, we use:
```sql
COALESCE(b.bureau_loan_count, 0) AS bureau_loan_count
```

**Why 0 instead of NULL:**

| Scenario | NULL Interpretation | 0 Interpretation |
|----------|--------------------|--------------------|
| No bureau record | "No data" (uncertain) | "No previous loans" (certain) |

**For counts:** 0 is more interpretable (no loans = 0, not "unknown")

**For averages/percentages:** NULL can remain NULL (handled by XGBoost)

---

## 5. Impact on Model Learning

### Feature Types Created

| Category | Examples | How Used |
|----------|----------|----------|
| Counts | bureau_loan_count, prev_app_count | Direct numeric input |
| Totals | bureau_total_debt, prev_app_amount_sum | Direct numeric input |
| Averages | bureau_credit_avg, prev_app_amount_mean | Direct numeric input |
| Ratios | bureau_active_pct, prev_app_refused_pct | Direct numeric input |
| Maxima | bureau_max_days_overdue, days_late_max | Direct numeric input |
| Binary flags | paid_late_count | Converted to rate |

### How XGBoost Uses These

**At each tree node, XGBoost:**

1. **Chooses a feature** (e.g., `bureau_active_pct`)
2. **Finds optimal split** (e.g., `pct <= 0.35` vs `pct > 0.35`)
3. **Computes gain** from split:

```
Gain = 0.5 * [G_left²/H_left + G_right²/H_right] - G_parent²/H_parent

Where:
G = sum of gradients (first derivative of loss)
H = sum of hessians (second derivative of loss)

For binary classification with log-loss:
gradient_i = p_i - y_i       (prediction - actual)
hessian_i = p_i * (1 - p_i)  (probability based)
```

**Example split:**
```
Feature: prev_app_refused_pct (proportion of refused applications)
Split: <= 0.25 vs > 0.25

Left child (low rejection rate):  default_rate = 5.2%
Right child (high rejection rate): default_rate = 18.7%

This split provides high information gain because
the two groups have very different outcomes.
```

---

## 6. Why These Specific Aggregations?

### Rationale Summary Table

| Table | Key Features | Business Logic |
|-------|---------------|----------------|
| Bureau | loan_count, total_debt, active_pct | External credit behavior |
| Bureau Balance | late_pct, status counts | Chronic payment issues |
| Previous App | refused_pct, approved_pct | Home Credit history |
| Installments | payment_diff, late_pct | Actual repayment behavior |
| POS Cash | status counts, counts | POS loan performance |
| Credit Card | balance, utilization | Credit card usage |

### Feature Engineering Principles Applied

1. **Capture diversity:** Different aspects of behavior
2. **Handle one-to-many:** Aggregate before join
3. **Create interpretable ratios:** Rates vs. raw counts
4. **Derive behavior indicators:** Payment patterns
5. **Handle nulls appropriately:** Certain zeros vs. unknown

---

## Summary

1. **Categorical handling:** XGBoost uses optimal partition encoding, learning category ordering from target variable
2. **Aggregations:** Transform one-to-many relationships into one-to-one, creating summary statistics
3. **Math:** XGBoost uses gradient-based splitting (gradient = ∂Loss/∂Prediction)
4. **Missing values:** Handled natively by sending missing values to optimal child branch
5. **Our approach:** COALESCE to 0 for counts (no record = 0), let XGBoost handle averages naturally
