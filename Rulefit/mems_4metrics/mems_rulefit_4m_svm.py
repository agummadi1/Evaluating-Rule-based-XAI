import pandas as pd
import numpy as np
import time
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from rulefit import RuleFit

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ⏱️ Start timer
start_time = time.time()

# Set your sample size
samples = 10000  # change to 500, 2500, or 10000

# Load dataset
df = pd.read_csv('/home/gummadi/mems_dataset.csv')
df = df[['x', 'label']]

# Analyze class counts
label_counts = df['label'].value_counts()
min_class_size = label_counts.min()
num_classes = label_counts.shape[0]

# Determine how many samples to take per class
test_ratio = 0.2
min_required_train_per_class = 10
min_required_total_per_class = int(min_required_train_per_class / (1 - test_ratio))

# Adjust max per class based on min required + available data
max_per_class = min(
    max(samples // num_classes, min_required_total_per_class),
    min_class_size
)

if max_per_class * num_classes < samples:
    print(f"⚠️ Adjusted total samples to {max_per_class * num_classes} due to class distribution.")
samples = max_per_class * num_classes

# ⚖️ Balanced sampling per class
df_sampled = (
    df.groupby('label', group_keys=False)
    .apply(lambda x: x.sample(n=max_per_class, random_state=42))
    .reset_index(drop=True)
)

# Features and labels
features = ['x']
X = df_sampled[features]
y = df_sampled['label']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_ratio, random_state=42, stratify=y
)

# 👨‍🏫 Choose SVM kernel based on sample size
if samples < 500:
    kernel_choice = 'linear'
else:
    kernel_choice = 'rbf'

# 🧠 Train SVM
svm_clf = SVC(kernel=kernel_choice, probability=True, class_weight='balanced', random_state=42)
svm_clf.fit(X_train, y_train)

# Predict + accuracy
y_pred_svm = svm_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_svm)
print("✅ SVM predicted label distribution:", dict(Counter(svm_clf.predict(X_train))))

# Use predicted labels for RuleFit explanation
svm_preds_train = svm_clf.predict(X_train)
unique_pred_classes = np.unique(svm_preds_train)

# Proceed only if SVM output is valid
if len(unique_pred_classes) >= 2:
    rf = RuleFit(tree_size=4, max_rules=2000, memory_par=0.01, rfmode='classification')
    rf.fit(X_train.values, svm_preds_train, feature_names=features)

    # Rule coverage
    try:
        rule_preds = rf.transform(X_test.values)
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
output_file = f"/home/gummadi/spring25/new_methods/rulefit_4m_svm/mems_rulefit_4m_svm.txt"
with open(output_file, 'a') as f:
    f.write("📊 Post-Hoc RuleFit Explanation for SVM Model\n\n")
    f.write(f"Total Samples Used: {samples} (≈ {max_per_class} per class)\n")
    f.write(f"Number of Classes: {num_classes}\n")
    f.write(f"SVM Kernel Used: {kernel_choice}\n")
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
