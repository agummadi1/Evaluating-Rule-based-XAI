import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from rulefit import RuleFit

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ⏱️ Start timer
start_time = time.time()

# Total number of samples to use
samples = 10000

# Load dataset
df = pd.read_csv("/home/gummadi/device7_top_20_features.csv")

# Ensure we don’t exceed available rows
if samples > len(df):
    raise ValueError(f"Requested {samples} samples, but dataset only has {len(df)} rows.")

# Stratified sampling to maintain class balance
df_balanced = df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(frac=samples / len(df), random_state=42)
).reset_index(drop=True)

# Feature and Target Separation
selected_features = [
    "HH_L1_magnitude", 
    "HH_L1_weight", 
    "HH_L3_magnitude",
    "HH_L5_magnitude", 
    "HH_L3_weight",
    "HH_L1_mean",
    "HH_L5_weight",
    "HH_L3_mean",
    "HH_L1_std",
    "HH_L5_mean",
    "HH_L3_std",
    "HH_L1_radius",
    "HH_L5_std",
    "HH_L3_pcc",
    "HH_L3_radius",
    "HH_L5_pcc",
    "HH_L5_radius",
    "HpHp_L0.01_pcc",
    "HH_L5_covariance", 
    "HH_L3_covariance"
]

X = df_balanced[selected_features]
y = df_balanced['label'].values
feature_names = selected_features


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train RuleFit
rf = RuleFit()
rf.fit(X_train.values, y_train, feature_names=feature_names)

#Predict and evaluate
y_pred = rf.predict(X_test.values).round().astype(int)
accuracy = accuracy_score(y_test, y_pred)

# Rule coverage
rule_preds = rf.transform(X_test.values)
avg_coverage = np.mean(rule_preds.sum(axis=1)) / rule_preds.shape[1]

# Get rules with non-zero coefficients
rules = rf.get_rules()
rules = rules[rules.coef != 0].copy()

# Extract feature name
rules['feature'] = rules['rule'].astype(str).str.extract(r'^([a-zA-Z_][a-zA-Z0-9_]*)')
rules = rules.dropna(subset=['feature'])

# Compute importance
importances = (
    rules.groupby('feature')['coef']
    .sum()
    .abs()
    .sort_values(ascending=False)
)

# Sparsity Values Calculation
thresholds = np.linspace(0, 1, 10)
sparsity_values = [
    sum(abs(importances) < t) / len(importances) for t in thresholds
]

# ⏱️ End timer
end_time = time.time()
elapsed_seconds = end_time - start_time
elapsed_hours = elapsed_seconds / 3600

# Save results
output_file = f"/home/gummadi/spring25/new_methods/rulefit_4metrics/d7_rulefit_4m_dt.txt"
with open(output_file, 'a') as f:
    f.write("📊 Overall RuleFit Model Metrics (Multiclass - Device7 Top 20 Features)\n\n")
    f.write(f"Total Samples Used: {samples}\n")
    f.write(f"Accuracy: {accuracy:.3f}\n")
    f.write(f"Efficiency (Execution Time): {elapsed_seconds:.2f} seconds / {elapsed_hours:.4f} hours\n")
    f.write(f"Avg Rule Coverage: {avg_coverage:.3f}\n\n")
    f.write("Feature Importance (Descending):\n")
    for feat, score in importances.items():
        f.write(f"{feat}: {score:.4f}\n")
    f.write("\nSparsity Values:\n")
    for t, s in zip(thresholds, sparsity_values):
        f.write(f"{t:.1f} : {s}\n")

print(f"✅ Results saved to {output_file}")
print(f"⏱️ Total Execution Time: {elapsed_seconds:.2f} seconds ({elapsed_hours:.4f} hours)")
print(f"📌 Total Samples Used: {samples}")
