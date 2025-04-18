import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from rulefit import RuleFit

# Load dataset
df = pd.read_csv('/home/gummadi/mems_dataset.csv')

# Use only relevant columns
df = df[['x', 'y', 'z', 'label']]

# Balance the dataset
min_samples = df['label'].value_counts().min()
df_balanced = df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=min_samples, random_state=42)
).reset_index(drop=True)

# Set features and target
features = ['x', 'y', 'z']
X = df_balanced[features]
y = df_balanced['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Class labels (you can rename if needed)
class_labels = {
    1: "Normal",
    2: "Near-failure",
    3: "Failure"
}

# Train separate RuleFit for each class and extract precision-evaluated rules
for class_label, class_name in class_labels.items():
    print(f"\n🔍 Training RuleFit for {class_name}...")

    # One-vs-Rest binary target
    y_train_binary = (y_train == class_label).astype(int)
    y_test_binary = (y_test == class_label).astype(int)

    # Train RuleFit
    rf = RuleFit(random_state=42)
    rf.fit(X_train_scaled, y_train_binary, feature_names=features)

    # Get non-zero rules
    rules = rf.get_rules()
    rules = rules[(rules.coef != 0) & (rules.rule != 'nan')]

    if rules.empty:
        print(f"No rules extracted for {class_name}")
        continue

    # Apply rules to test set
    rule_preds = rf.transform(X_test_scaled)

    # Calculate precision and coverage
    precisions = []
    rule_texts = []
    supports = []

    for i, row in rules.reset_index(drop=True).iterrows():
        rule_name = row['rule']
        pred = rule_preds[:, i]

        fired = pred == 1
        if fired.sum() == 0:
            continue

        tp = ((pred == 1) & (y_test_binary == 1)).sum()
        fp = ((pred == 1) & (y_test_binary == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        coverage = fired.mean()

        precisions.append(precision)
        rule_texts.append(rule_name)
        supports.append(coverage)

    # Top 3 rules by coverage
    rule_df = pd.DataFrame({
        'rule': rule_texts,
        'precision': precisions,
        'coverage': supports
    }).sort_values(by='coverage', ascending=False).head(3)

    # Save results
    output_lines = [f"RuleFit Rules for {class_name}:"]
    for _, row in rule_df.iterrows():
        output_lines.append(f"{row['rule']}")
    output_lines.append(f"\nAverage Precision: {rule_df['precision'].mean():.2f}")
    output_lines.append(f"Average Coverage: {rule_df['coverage'].mean():.2f}")

    output_file = f"/home/gummadi/spring25/new_methods/working_rulefit/mems_rulefit2_rules_for_class_{class_label}.txt"
    with open(output_file, 'w') as f:
        for line in output_lines:
            f.write(line + "\n")

    print(f"✅ Saved precision-evaluated rules for {class_name} to {output_file}")
