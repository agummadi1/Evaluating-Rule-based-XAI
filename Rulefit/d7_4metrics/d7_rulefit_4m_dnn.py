import pandas as pd
import numpy as np
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from rulefit import RuleFit
import warnings, os

warnings.filterwarnings("ignore")

# ⏱️ Start timer
start_time = time.time()
samples = 10000

# Load dataset
df = pd.read_csv("/home/gummadi/device7_top_20_features.csv")
selected_features = [
    #"HH_L1_magnitude", 
    "HH_L1_weight", 
    #"HH_L3_magnitude", 
    #"HH_L5_magnitude",
    #"HH_L3_weight", 
    #"HH_L1_mean", 
    "HH_L5_weight", 
    #"HH_L3_mean", 
    #"HH_L1_std",
    "HH_L5_mean", "HH_L3_std", 
    #"HH_L1_radius", 
    #"HH_L5_std", 
    #"HH_L3_pcc",
    #"HH_L3_radius", #"HH_L5_pcc", "HH_L5_radius", 
     "HpHp_L0.01_pcc"
    #"HH_L5_covariance", "HH_L3_covariance"
]
X_full = df[selected_features]
y_full = df['label']

# Balanced sampling
class_counts = y_full.value_counts()
num_classes = len(class_counts)

samples = (samples // num_classes) * num_classes  # auto-adjust to nearest divisible count

if samples % num_classes != 0:
    raise ValueError(f"Sample count ({samples}) must be divisible by number of classes ({num_classes}).")

n_per_class = samples // num_classes
min_class_size = class_counts.min()
if n_per_class > min_class_size:
    raise ValueError(f"Requested {n_per_class} per class, but some classes have only {min_class_size} samples.")

df['label'] = y_full
df_sampled = df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=n_per_class, random_state=42)
).reset_index(drop=True)

X = df_sampled[selected_features]
y_raw = df_sampled['label']
class_labels = sorted(y_raw.unique())
label_to_index = {label: idx for idx, label in enumerate(class_labels)}
y = y_raw.map(label_to_index)
num_classes = len(class_labels)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# DNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, verbose=0)

y_test_pred = np.argmax(model.predict(X_test_scaled), axis=1)
dnn_accuracy = accuracy_score(y_test, y_test_pred)

y_train_pred = np.argmax(model.predict(X_train_scaled), axis=1)

# RuleFit
rf = RuleFit(tree_size=4, max_rules=2000, rfmode='classification', memory_par=0.01, exp_rand_tree_size=True)
rf.fit(X_train_scaled, y_train_pred, feature_names=selected_features)

y_rf_pred = rf.predict(X_test_scaled).round().astype(int)
rulefit_accuracy = accuracy_score(y_test_pred, y_rf_pred)

rule_preds = rf.transform(X_test_scaled)
avg_coverage = np.mean(rule_preds.sum(axis=1)) / rule_preds.shape[1] if rule_preds.shape[1] > 0 else 0.0

rules = rf.get_rules()
rules = rules[rules.coef != 0].copy()
rules['feature'] = rules['rule'].astype(str).str.extract(r'^([a-zA-Z_][a-zA-Z0-9_]*)')
rules = rules.dropna(subset=['feature'])

importances = (
    rules.groupby('feature')['coef']
    .sum()
    .abs()
    .sort_values(ascending=False)
)

thresholds = np.linspace(0, 1, 10)
importances_norm = importances / importances.max()
sparsity_values = [
    sum(abs(importances_norm) < t) / len(importances_norm) for t in thresholds
]

# ⏱️ End timer
end_time = time.time()
elapsed_seconds = end_time - start_time
elapsed_hours = elapsed_seconds / 3600

# Output
output_file = f"/home/gummadi/spring25/new_methods/rulefit_4m_dnn/d7_rulefit_4m_dnn.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

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
