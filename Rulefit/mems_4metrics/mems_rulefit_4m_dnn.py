import pandas as pd
import numpy as np
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from rulefit import RuleFit
import warnings

warnings.filterwarnings("ignore")

# ⏱️ Start timer
start_time = time.time()

# Number of total samples to use
samples = 1000

# Load dataset
df = pd.read_csv('/home/gummadi/mems_dataset.csv')
df = df[['x', 'label']]

# Validate sample count
if samples > len(df):
    raise ValueError(f"Requested {samples} samples, but dataset only has {len(df)} rows.")

# Stratified sampling to keep class balance
# df_sampled = df.groupby('label', group_keys=False).apply(
#     lambda x: x.sample(frac=samples / len(df), random_state=42)
# ).reset_index(drop=True)

# Balanced sampling: equal number of samples per class
class_counts = df['label'].value_counts()
num_classes = len(class_counts)

if samples % num_classes != 0:
    raise ValueError(f"Sample count ({samples}) must be divisible by number of classes ({num_classes}).")

n_per_class = samples // num_classes
min_class_size = class_counts.min()

if n_per_class > min_class_size:
    raise ValueError(f"Requested {n_per_class} per class, but some classes have only {min_class_size} samples.")

df_sampled = (
    df.groupby('label', group_keys=False)
    .apply(lambda x: x.sample(n=n_per_class, random_state=42))
    .reset_index(drop=True)
)


# Features and target
features = ['x']
X = df_sampled[features]
y_raw = df_sampled['label']

# Encode labels (required for softmax output)
class_labels = sorted(y_raw.unique())
label_to_index = {label: idx for idx, label in enumerate(class_labels)}
y = y_raw.map(label_to_index)
num_classes = len(class_labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✅ Train DNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, verbose=0)

# 📊 DNN Evaluation
y_test_probs = model.predict(X_test_scaled)
y_test_pred = np.argmax(y_test_probs, axis=1)
dnn_accuracy = accuracy_score(y_test, y_test_pred)

# Get predictions on train set for RuleFit
y_train_probs = model.predict(X_train_scaled)
y_train_pred = np.argmax(y_train_probs, axis=1)

# ✅ RuleFit (Post-hoc mimic model)
rf = RuleFit(
    tree_size=4,
    max_rules=2000,
    rfmode='classification',
    memory_par=0.01,
    exp_rand_tree_size=True
)
rf.fit(X_train_scaled, y_train_pred, feature_names=features)

# RuleFit mimic predictions
y_rf_pred = rf.predict(X_test_scaled).round().astype(int)
rulefit_accuracy = accuracy_score(y_test_pred, y_rf_pred)

# Rule coverage
rule_preds = rf.transform(X_test_scaled)
avg_coverage = np.mean(rule_preds.sum(axis=1)) / rule_preds.shape[1] if rule_preds.shape[1] > 0 else 0.0

# Get rules with non-zero coefficients
rules = rf.get_rules()
rules = rules[rules.coef != 0].copy()
rules['feature'] = rules['rule'].astype(str).str.extract(r'^([a-zA-Z_][a-zA-Z0-9_]*)')
rules = rules.dropna(subset=['feature'])

# Compute importance
importances = (
    rules.groupby('feature')['coef']
    .sum()
    .abs()
    .sort_values(ascending=False)
)

# Sparsity values
thresholds = np.linspace(0, 1, 10)

importances_norm = importances / importances.max()
sparsity_values = [
    sum(abs(importances_norm) < t) / len(importances_norm) for t in thresholds
]

# sparsity_values = [
#     sum(abs(importances) < t) / len(importances) for t in thresholds
# ] if not importances.empty else [1.0] * len(thresholds)

# ⏱️ End timer
end_time = time.time()
elapsed_seconds = end_time - start_time
elapsed_hours = elapsed_seconds / 3600

# Save results
output_file = f"/home/gummadi/spring25/new_methods/rulefit_4m_dnn/mems_rulefit_4m_dnn.txt"
with open(output_file, 'a') as f:
    f.write("📊 RuleFit Post-Hoc Model Metrics (DNN Mimic - Multiclass)\n\n")
    f.write(f"Total Samples Used: {samples}\n")
    f.write(f"DNN Accuracy: {dnn_accuracy:.3f}\n")
    f.write(f"RuleFit Mimic Accuracy: {rulefit_accuracy:.3f}\n")
    f.write(f"Efficiency (Execution Time): {elapsed_seconds:.2f} seconds / {elapsed_hours:.4f} hours\n")
    f.write(f"Avg Rule Coverage: {avg_coverage:.3f}\n\n")
    f.write("Feature Importance (Descending):\n")
    for feat, score in importances.items():
        f.write(f"{feat}: {score:.4f}\n")
    f.write("\nSparsity Values:\n")
    for t, s in zip(thresholds, sparsity_values):
        f.write(f"{t:.1f} : {s:.3f}\n")

print(f"\n✅ Results saved to {output_file}")
print(f"📈 DNN Accuracy: {dnn_accuracy:.3f}")
print(f"🧠 RuleFit Mimic Accuracy: {rulefit_accuracy:.3f}")
print(f"⏱️ Execution Time: {elapsed_seconds:.2f} seconds ({elapsed_hours:.4f} hours)")
