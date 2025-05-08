import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from rulefit import RuleFit

# Load and balance dataset
df = pd.read_csv('/home/gummadi/mems_dataset.csv')
df = df[['x', 'y', 'z', 'label']]
target_total = 1000
num_classes = df['label'].nunique()
samples_per_class = target_total // num_classes

df_balanced = df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=min(samples_per_class, len(x)), random_state=42)
).reset_index(drop=True)

print("🔢 Sample count per class after balancing:")
print(df_balanced['label'].value_counts())

# Feature-target split
features = ['x', 'y', 'z']
X = df_balanced[features]
y = df_balanced['label']
y_encoded = LabelEncoder().fit_transform(y)
class_labels = {
    0: "Normal",
    1: "Near-failure",
    2: "Failure"
}

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# DNN model for classification
n_classes = len(np.unique(y_train))
dnn = Sequential([
    Dense(32, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(n_classes, activation='softmax')
])
dnn.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
dnn.fit(X_train_scaled, to_categorical(y_train), epochs=50, verbose=0, batch_size=32)

# Get predicted class labels from DNN for RuleFit training
dnn_preds_train = np.argmax(dnn.predict(X_train_scaled), axis=1)
dnn_preds_test = np.argmax(dnn.predict(X_test_scaled), axis=1)

# Train RuleFit on DNN predictions
rf = RuleFit(random_state=42)
rf.fit(X_train_scaled, dnn_preds_train, feature_names=features)

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
    fired_labels = dnn_preds_test[fired_idx]

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
output_lines = ["Top 5 RuleFit Rules (All Classes Combined) - Trained on DNN\n"]
output_lines.append(f"{'Rule':<60} | {'Class':<15} | {'Precision':<9} | {'Coverage'}")
output_lines.append("-" * 100)

for _, row in top_rules.iterrows():
    output_lines.append(
        f"{row['rule']:<60} | {row['class_name']:<15} | {row['precision']:.2f}      | {row['coverage']:.2f}"
    )

# Summary
output_lines.append("\n\nSummary (across all rules):")
output_lines.append(f"Total rules evaluated: {len(results_df)}")
output_lines.append(f"Average Precision: {results_df['precision'].mean():.2f}")
output_lines.append(f"Average Coverage: {results_df['coverage'].mean():.2f}")

# Write to file
output_file = "/home/gummadi/spring25/extras/prec_cov_rulefit/mems/dnn2.txt"
with open(output_file, 'a') as f:
    for line in output_lines:
        f.write(line + "\n")

print(f"✅ DNN-based RuleFit summary saved to {output_file}")
