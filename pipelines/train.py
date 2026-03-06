import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Load the fully processed Parquet file
print("Loading data...")
df = pd.read_parquet("data/processed/application_features.parquet")

df["TARGET"] = pd.to_numeric(df["TARGET"], errors="coerce")

for col in df.select_dtypes(include=["object", "str"]).columns:
    df[col] = df[col].astype("category")

# 2. The Split: Separate Train and Test using your DuckDB flag
print("Splitting data...")
train_data = df[df["is_train"] == 1].copy()
test_data = df[df["is_train"] == 0].copy()

# Drop utility columns that aren't actual predictive features
cols_to_drop = ["SK_ID_CURR", "TARGET", "is_train"]

X = train_data.drop(columns=cols_to_drop)
y = train_data["TARGET"].astype(int)  # Ensure it's explicitly an integer for XGBoost

# The competition holdout set (what you will eventually predict on)
X_competition = test_data.drop(columns=cols_to_drop)


# 3. Create a local validation set to test our model before doing the final predictions
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Handle the Imbalance (The 11.4 to 1 ratio your script found)
# Formula: count(negative examples) / count(positive examples)
imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum()

# 5. Initialize and Train XGBoost
print(f"\nTraining XGBoost with scale_pos_weight={imbalance_ratio:.2f}...")
model = xgb.XGBClassifier(
    n_estimators=1000,  # Max number of trees to build
    learning_rate=0.05,
    max_depth=4,
    scale_pos_weight=imbalance_ratio,  # Forces the model to care about defaults
    enable_categorical=True,  # Lets XGBoost process strings without One-Hot Encoding
    eval_metric="auc",  # Optimize for Area Under the ROC Curve
    early_stopping_rounds=50,  # Stop if the validation score doesn't improve for 50 rounds
    tree_method="hist",  # Highly optimized algorithm for modern CPUs
    n_jobs=-1,  # Max out all available CPU cores
    min_child_weight=30,
    colsample_bytree=0.8,
    subsample=0.8,
)

# Train the model, watching the validation set to prevent overfitting
model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=50,  # Print an update every 50 trees
)

# 6. Score the model
# We use [:, 1] to get the probability of a default (Class 1) rather than just a 0 or 1 prediction
val_preds = model.predict_proba(X_val)[:, 1]
auc_score = roc_auc_score(y_val, val_preds)

print(f"\nFinal Validation ROC-AUC Score: {auc_score:.4f}")


print("\nGenerating Feature Importances...")
# 7. Extract and Plot Feature Importances
importance = model.feature_importances_
features = X_train.columns

# Create a clean DataFrame
fi_df = pd.DataFrame({"Feature": features, "Importance": importance})
fi_df = fi_df.sort_values(by="Importance", ascending=False).head(20)

# Generate the plot
plt.figure(figsize=(12, 8))
plt.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1], color="steelblue")
plt.title("Top 20 Strongest Predictors of Default (XGBoost)")
plt.xlabel("Relative Importance")
plt.tight_layout()

# Save it so you can open it on your Mac
plt.savefig("feature_importances.png")
print("Saved feature importance plot to feature_importances.png")

print("\nGenerating final predictions on the blind test set...")

# 1. Predict probabilities for the test set
# We use [:, 1] to grab the probability of class 1 (Default)
test_preds = model.predict_proba(X_competition)[:, 1]

# 2. Create the submission dataframe
# We need the SK_ID_CURR from the original test_data, and our new predictions
submission = pd.DataFrame({"SK_ID_CURR": test_data["SK_ID_CURR"].astype(int), "TARGET": test_preds})

# 3. Save to CSV
output_file = "data/processed/submission.csv"
submission.to_csv(output_file, index=False)

print(f"Successfully saved {len(submission):,} predictions to {output_file}")
print("Sample of final predictions:")
print(submission.head())

import shap
import matplotlib.pyplot as plt

print("\nGenerating SHAP explanations...")

# 1. Initialize the SHAP Explainer
# This algorithm reverse-engineers the XGBoost trees
explainer = shap.TreeExplainer(model)

# 2. Find the most extreme cases in your test set
high_risk_idx = test_preds.argmax()  # The person most likely to default
low_risk_idx = test_preds.argmin()  # The person least likely to default

# Get the actual applicant IDs for our records
high_risk_id = test_data.iloc[high_risk_idx]["SK_ID_CURR"]
low_risk_id = test_data.iloc[low_risk_idx]["SK_ID_CURR"]

# 3. Extract their specific feature rows
applicant_high = X_competition.iloc[[high_risk_idx]]
applicant_low = X_competition.iloc[[low_risk_idx]]

# 4. Calculate the SHAP values
shap_values_high = explainer(applicant_high)

# 5. Generate and save a Waterfall Plot for the High-Risk applicant
plt.figure(figsize=(12, 8))
# We pass shap_values_high[0] because we are only explaining one person
shap.plots.waterfall(shap_values_high[0], max_display=15, show=False)
plt.title(f"Why is Applicant {high_risk_id} HIGH risk? (Prob: {test_preds[high_risk_idx]:.2%})")
plt.tight_layout()
plt.savefig("shap_high_risk.png", bbox_inches="tight")
plt.show()
plt.close()

print(f"Saved SHAP explanation for High-Risk Applicant {high_risk_id} to shap_high_risk.png")

# Print SHAP values to console
print("\n=== SHAP Values for High-Risk Applicant ===")
print(f"Base value: {shap_values_high.base_values[0]:.4f}")
print(f"Prediction: {shap_values_high.base_values[0] + shap_values_high.values[0].sum():.4f}")
print("\nTop feature contributions:")
shap_df = pd.DataFrame(
    {
        "Feature": X_competition.columns,
        "Value": applicant_high.values[0],
        "SHAP": shap_values_high.values[0],
    }
)
shap_df["Abs_SHAP"] = shap_df["SHAP"].abs()
shap_df = shap_df.sort_values("Abs_SHAP", ascending=False).head(15)
for _, row in shap_df.iterrows():
    direction = "↑" if row["SHAP"] > 0 else "↓"
    print(f"  {row['Feature']}: {row['Value']} → {row['SHAP']:+.4f} {direction}")
