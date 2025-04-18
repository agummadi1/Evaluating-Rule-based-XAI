import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from rulefit import RuleFit

# Load dataset
df = pd.read_csv('/home/gummadi/device7_top_20_features.csv')

# Explicitly define features and target
feature_columns = [col for col in df.columns if col != 'label']
target = 'label'

# Balance dataset
min_samples = df[target].value_counts().min()
df_balanced = df.groupby(target, group_keys=False).apply(
    lambda x: x.sample(n=min_samples, random_state=42)
).reset_index(drop=True)

X = df_balanced[feature_columns]
y = df_balanced[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit RuleFit
rulefit = RuleFit(random_state=42)
rulefit.fit(X_train_scaled, y_train, feature_names=feature_columns)

# Extract rules
rules = rulefit.get_rules()
rules = rules[rules.coef != 0]

# Process rules per class
class_labels = {
    1: "Class 1", 2: "Class 2", 3: "Class 3",
    4: "Class 4", 5: "Class 5", 6: "Class 6"
}

from sklearn.metrics import precision_score

# For each class, train RuleFit and extract precision per rule
for class_label, class_name in class_labels.items():
    print(f"\n🔍 Training RuleFit for {class_name}...")

    # One-vs-Rest target
    y_train_binary = (y_train == class_label).astype(int)
    y_test_binary = (y_test == class_label).astype(int)

    # Train RuleFit
    rf = RuleFit(random_state=42)
    rf.fit(X_train_scaled, y_train_binary, feature_names=feature_columns)

    # Get non-zero rules
    rules = rf.get_rules()
    rules = rules[(rules.coef != 0) & (rules.rule != 'nan')]

    if rules.empty:
        print(f"No rules extracted for {class_name}")
        continue

    # Evaluate rules on test set
    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_columns)
    rule_preds = rf.transform(X_test_scaled)

    precisions = []
    rule_texts = []
    supports = []

    # Loop over rules with reset index
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

    
    # Put results in DataFrame
    rule_df = pd.DataFrame({
        'rule': rule_texts,
        'precision': precisions,
        'coverage': supports
    }).sort_values(by='coverage', ascending=False).head(5)

    # Save to file
    output_lines = [f"RuleFit Rules for {class_name}:"]
    for _, row in rule_df.iterrows():
        output_lines.append(str(row['rule']))
    output_lines.append(f"\nAverage Precision: {rule_df['precision'].mean():.2f}")
    output_lines.append(f"Average Coverage: {rule_df['coverage'].mean():.2f}")

    output_file = f"/home/gummadi/spring25/new_methods/working_rulefit/d7_rules_for_class_{class_label}.txt"
    with open(output_file, 'w') as f:
        for line in output_lines:
            f.write(line + "\n")

    print(f"✅ Saved precision-evaluated rules for {class_name} to {output_file}")