import pandas as pd
import numpy as np
import time
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from rulefit import RuleFit

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ⏱️ Start timer
start_time = time.time()

# Set sample size
samples = 10000

# Load dataset
df = pd.read_csv("/home/gummadi/device7_top_20_features.csv")

selected_features = [
    #"HH_L1_magnitude", 
    #"HH_L1_weight", 
    "HH_L3_magnitude",
    #"HH_L5_magnitude", 
    #"HH_L3_weight",
    #"HH_L1_mean",
    "HH_L5_weight",
    # "HH_L3_mean",
    # "HH_L1_std",
    "HH_L5_mean",
    #"HH_L3_std",
    "HH_L1_radius",
    "HH_L5_std",
    #"HH_L3_pcc",
    #"HH_L3_radius",
    #"HH_L5_pcc",
    #"HH_L5_radius",
    #"HpHp_L0.01_pcc",
    #"HH_L5_covariance", 
    #"HH_L3_covariance"
]

X_full = df[selected_features]
y_full = df['label']
feature_names = selected_features

# Analyze class distribution
label_counts = y_full.value_counts()
min_class_size = label_counts.min()
num_classes = label_counts.shape[0]

# Determine samples per class
test_ratio = 0.2
min_required_train_per_class = 10
min_required_total_per_class = int(min_required_train_per_class / (1 - test_ratio))

max_per_class = min(
    max(samples // num_classes, min_required_total_per_class),
    min_class_size
)

if max_per_class * num_classes < samples:
    print(f"⚠️ Adjusted total samples to {max_per_class * num_classes} due to class distribution.")
samples = max_per_class * num_classes

# ⚖️ Balanced sampling per class
df_balanced = (
    df.groupby('label', group_keys=False)
    .apply(lambda x: x.sample(n=max_per_class, random_state=42))
    .reset_index(drop=True)
)

X = df_balanced[selected_features]
y = df_balanced['label']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_ratio, random_state=42, stratify=y
)

# 📊 Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔍 Tune SVM with GridSearch
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(probability=True, class_weight='balanced'), param_grid, cv=3, n_jobs=-1)
grid.fit(X_train_scaled, y_train)
svm_clf = grid.best_estimator_
print(f"🔧 Best SVM parameters: {grid.best_params_}")

# Evaluate
y_pred_svm = svm_clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred_svm)
print("✅ SVM predicted class distribution:", dict(Counter(svm_clf.predict(X_train_scaled))))
print("📊 Classification Report:\n", classification_report(y_test, y_pred_svm))

# Predict on train for RuleFit
svm_preds_train = svm_clf.predict(X_train_scaled)
predicted_train_class_counts = dict(Counter(svm_preds_train))
unique_pred_classes = np.unique(svm_preds_train)
print(f"✅ SVM predicted class distribution on training set: {predicted_train_class_counts}")

# Train RuleFit if SVM output is valid
if len(unique_pred_classes) >= 2:
    try:
        rf = RuleFit(tree_size=4, max_rules=2000, memory_par=0.01,
                     rfmode='classification', cv=2)  # ✅ avoid crash on edge cases
        rf.fit(X_train_scaled, svm_preds_train, feature_names=feature_names)

        # Rule coverage
        try:
            rule_preds = rf.transform(X_test_scaled)
            if len(rule_preds.shape) == 1:
                avg_coverage = 0.0
            else:
                avg_coverage = np.mean(rule_preds.sum(axis=1)) / rule_preds.shape[1]
        except Exception:
            avg_coverage = 0.0

        # Rule extraction
        rules = rf.get_rules()
        rules = rules[rules.coef != 0].copy()
        print(f"🧱 Extracted Rules: {len(rules)}")

        if not rules.empty:
            rules['feature'] = rules['rule'].astype(str).str.extract(r'^([a-zA-Z_][a-zA-Z0-9_]*)')
            rules = rules.dropna(subset=['feature'])

            importances = (
                rules.groupby('feature')['coef']
                .sum()
                .abs()
                .sort_values(ascending=False)
            )

            thresholds = np.linspace(0, 1, 10)
            sparsity_values = [
                sum(abs(importances) < t) / len(importances) for t in thresholds
            ]
        else:
            importances = pd.Series(dtype=float)
            thresholds = np.linspace(0, 1, 10)
            sparsity_values = [1.0] * len(thresholds)
    except ValueError as ve:
        print(f"❌ RuleFit training failed: {ve}")
        avg_coverage = 0.0
        rules = pd.DataFrame()
        importances = pd.Series(dtype=float)
        thresholds = np.linspace(0, 1, 10)
        sparsity_values = [1.0] * len(thresholds)
else:
    print(f"⚠️ RuleFit skipped: SVM predicted only one class → {unique_pred_classes}")
    avg_coverage = 0.0
    rules = pd.DataFrame()
    importances = pd.Series(dtype=float)
    thresholds = np.linspace(0, 1, 10)
    sparsity_values = [1.0] * len(thresholds)

# ⏱️ End timer
end_time = time.time()
elapsed_seconds = end_time - start_time
elapsed_hours = elapsed_seconds / 3600

# Save results
output_file = "/home/gummadi/spring25/new_methods/rulefit_4m_svm/d7_rulefit_4m_svm.txt"
with open(output_file, 'a') as f:
    f.write("📊 Post-Hoc RuleFit Explanation for SVM Model (device7 dataset)\n\n")
    f.write(f"Total Samples Used: {samples} (≈ {max_per_class} per class)\n")
    f.write(f"Number of Classes: {num_classes}\n")
    f.write(f"Best SVM Params: {grid.best_params_}\n")
    f.write(f"SVM Accuracy: {accuracy:.3f}\n")
    f.write(f"Execution Time: {elapsed_seconds:.2f} seconds / {elapsed_hours:.4f} hours\n")
    f.write(f"Avg Rule Coverage: {avg_coverage:.3f}\n")
    f.write(f"Extracted Rules: {len(rules)}\n\n")

    f.write("Feature Importance (Descending):\n")
    if not importances.empty:
        for feat, score in importances.items():
            f.write(f"{feat}: {score:.4f}\n")
    else:
        f.write("No important rules were extracted.\n")

    f.write("\nSparsity Values:\n")
    for t, s in zip(thresholds, sparsity_values):
        f.write(f"{t:.1f} : {s:.3f}\n")

print(f"✅ Results saved to {output_file}")
print(f"⏱️ Total Execution Time: {elapsed_seconds:.2f} seconds ({elapsed_hours:.4f} hours)")
print(f"📌 Total Samples Used: {samples} (≈ {max_per_class} per class)")
