import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from rulefit import RuleFit

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ⏱️ Start timer
start_time = time.time()

# Load dataset
df = pd.read_csv("/home/gummadi/device7_top_20_features.csv")

# Selected Features
selected_features = [
    #"HH_L1_magnitude", 
    #"HH_L1_weight", 
    #"HH_L3_magnitude",
    #"HH_L5_magnitude", 
    "HH_L3_weight",
    #"HH_L1_mean",
    #"HH_L5_weight",
    #"HH_L3_mean",
    "HH_L1_std",
    #"HH_L5_mean",
    "HH_L3_std",
    #"HH_L1_radius",
    "HH_L5_std",
    #"HH_L3_pcc",
    #"HH_L3_radius",
    #"HH_L5_pcc",
    "HH_L5_radius",
    #"HpHp_L0.01_pcc",
    #"HH_L5_covariance", 
    #"HH_L3_covariance"
]

# Balance classes using stratified sampling
samples = 10000
if samples > len(df):
    raise ValueError(f"Requested {samples} samples, but dataset only has {len(df)} rows.")

df_balanced = df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(frac=samples / len(df), random_state=42)
).reset_index(drop=True)

X = df_balanced[selected_features]
y = df_balanced['label'].values
feature_names = selected_features

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Train Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
rf_preds = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)

# ✅ Train RuleFit
rf_model = RuleFit(tree_size=4, rfmode='classification', max_rules=2000)
rf_model.fit(X_train.values, y_train, feature_names=feature_names)

rulefit_preds = rf_model.predict(X_test.values).round().astype(int)
rulefit_accuracy = accuracy_score(y_test, rulefit_preds)

# Rule coverage
rule_preds = rf_model.transform(X_test.values)
avg_coverage = np.mean(rule_preds.sum(axis=1)) / rule_preds.shape[1]

# Rule importance
rules = rf_model.get_rules()
rules = rules[rules.coef != 0].copy()
rules['feature'] = rules['rule'].astype(str).str.extract(r'^([a-zA-Z_][a-zA-Z0-9_]*)')
rules = rules.dropna(subset=['feature'])
importances = (
    rules.groupby('feature')['coef']
    .sum()
    .abs()
    .sort_values(ascending=False)
)

# Sparsity values
thresholds = np.linspace(0, 1, 10)
sparsity_values = [
    sum(abs(importances) < t) / len(importances) for t in thresholds
]

# ⏱️ End timer
end_time = time.time()
elapsed_seconds = end_time - start_time
elapsed_hours = elapsed_seconds / 3600

# Save results
output_file = "/home/gummadi/spring25/new_methods/rulefit_4m_rf/d7_rulefit_4m_rf.txt"
with open(output_file, 'a') as f:
    f.write("📊 Model Evaluation: Random Forest vs RuleFit (Device7)\n\n")
    f.write(f"Total Samples Used: {samples}\n")
    f.write(f"Random Forest Accuracy: {rf_accuracy:.3f}\n")
    f.write(f"RuleFit Accuracy: {rulefit_accuracy:.3f}\n")
    f.write(f"Execution Time: {elapsed_seconds:.2f} seconds / {elapsed_hours:.4f} hours\n")
    f.write(f"Avg Rule Coverage (RuleFit): {avg_coverage:.3f}\n\n")
    f.write("Feature Importance (RuleFit - Descending):\n")
    for feat, score in importances.items():
        f.write(f"{feat}: {score:.4f}\n")
    f.write("\nSparsity Values (RuleFit):\n")
    for t, s in zip(thresholds, sparsity_values):
        f.write(f"{t:.1f} : {s:.3f}\n")

print(f"✅ Results saved to {output_file}")
print(f"⏱️ Total Execution Time: {elapsed_seconds:.2f} seconds ({elapsed_hours:.4f} hours)")
print(f"📌 Total Samples Used: {samples}")
