import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from rulefit import RuleFit

# Load dataset
df = pd.read_csv('/home/gummadi/device7_top_20_features.csv')
feature_cols = [col for col in df.columns if col != 'label']

# Balance to ~1000 total instances
target_total = 1000
num_classes = df['label'].nunique()
samples_per_class = target_total // num_classes

df_balanced = df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=min(samples_per_class, len(x)), random_state=42)
).reset_index(drop=True)

print("🔢 Sample count per class after balancing:")
print(df_balanced['label'].value_counts())

# Feature-target split
X = df_balanced[feature_cols]
y = df_balanced['label']
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_labels = {i: f"Class {label}" for i, label in enumerate(label_encoder.classes_)}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM classifier
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)

# Get predicted labels
svm_preds_train = svm.predict(X_train_scaled)
svm_preds_test = svm.predict(X_test_scaled)

# RuleFit on SVM predictions
rf = RuleFit(random_state=42)
rf.fit(X_train_scaled, svm_preds_train, feature_names=feature_cols)

# Extract non-zero rules
rules = rf.get_rules()
rules = rules[(rules.coef != 0) & (rules.rule != 'nan')].reset_index(drop=True)

# Apply rules to test set
rule_preds = rf.transform(X_test_scaled)

# Evaluate rules
results = []
for i, row in rules.iterrows():
    rule_text = row['rule']
    pred = rule_preds[:, i]
    if pred.sum() == 0:
        continue

    fired_idx = np.where(pred == 1)[0]
    fired_labels = svm_preds_test[fired_idx]

    support = len(fired_idx) / len(y_test)
    class_counts = pd.Series(fired_labels).value_counts(normalize=True)
    best_class = class_counts.idxmax()
    precision = class_counts.max()

    results.append({
        'rule': rule_text,
        'best_class': best_class,
        'class_name': class_labels[best_class],
        'precision': precision,
        'coverage': support
    })

# Results DataFrame
results_df = pd.DataFrame(results)
top_rules = results_df.sort_values(by='coverage', ascending=False).head(5)

# Output lines
output_lines = ["Top 5 RuleFit Rules (All Classes Combined) - Trained on SVM (Device7 Dataset)\n"]
output_lines.append(f"{'Rule':<90} | {'Class':<12} | {'Precision':<9} | {'Coverage'}")
output_lines.append("-" * 130)

for _, row in top_rules.iterrows():
    output_lines.append(
        f"{row['rule']:<90} | {row['class_name']:<12} | {row['precision']:.2f}      | {row['coverage']:.2f}"
    )

# Summary
output_lines.append("\n\nSummary (across all rules):")
output_lines.append(f"Total rules evaluated: {len(results_df)}")
output_lines.append(f"Average Precision: {results_df['precision'].mean():.2f}")
output_lines.append(f"Average Coverage: {results_df['coverage'].mean():.2f}")

# Save output
output_file = "/home/gummadi/spring25/extras/prec_cov_rulefit/d7/d7_rules_prec_cov_rulefit_svm.txt"
with open(output_file, 'a') as f:
    for line in output_lines:
        f.write(line + "\n")

print(f"✅ SVM-based RuleFit summary saved to {output_file}")
