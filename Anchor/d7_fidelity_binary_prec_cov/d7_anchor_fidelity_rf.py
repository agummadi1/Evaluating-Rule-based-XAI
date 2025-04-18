import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from alibi.explainers import AnchorTabular
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Output file
output_path = "/home/gummadi/spring25/week12/anchor/d7/d7_anchor_fidelity_rf.txt"

# Load and balance dataset
file_path = "/home/gummadi/device7_top_20_features.csv"
df = pd.read_csv(file_path)

min_samples = df['label'].value_counts().min()
df_balanced = df.groupby('label').apply(lambda x: x.sample(min_samples, random_state=42)).reset_index(drop=True)

# Feature and target separation
X = df_balanced.drop(columns=['label']).values
feature_names = df_balanced.drop(columns=['label']).columns.tolist()
y = LabelEncoder().fit_transform(df_balanced['label'].values)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Model: Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Anchor Explainer
explainer = AnchorTabular(model.predict, feature_names=feature_names)
explainer.fit(X_train, disc_perc=(5, 10, 25, 50, 75, 90, 95))

# Metric collection
fidelity_scores = []
precision_scores = []
coverage_scores = []

for idx in range(min(1000, len(X_test))):
    instance = X_test[idx].reshape(1, -1)
    try:
        explanation = explainer.explain(instance[0], threshold=0.85, n_covered_ex=20000)
        original_pred = model.predict(instance)[0]
        perturbed_samples = explanation.raw['instances']
        if len(perturbed_samples) == 0:
            continue

        pred_matches = model.predict(perturbed_samples) == original_pred
        fidelity = np.mean(pred_matches)

        fidelity_scores.append(fidelity)
        precision_scores.append(explanation.precision)
        coverage_scores.append(explanation.coverage)

    except Exception:
        continue

# Write results
with open(output_path, "a") as f:
    if fidelity_scores:
        overall_fidelity = round(np.mean(fidelity_scores), 4)
        avg_precision = round(np.mean(precision_scores), 4)
        avg_coverage = round(np.mean(coverage_scores), 4)

        message = (
            f"✅ Anchor Explanation Summary for Random Forest (Device7 Dataset)\n"
            f"Instances explained: {len(fidelity_scores)}\n\n"
            f"✅ Overall Fidelity: {overall_fidelity}\n"
            f"✅ Average Precision: {avg_precision}\n"
            f"✅ Average Coverage: {avg_coverage}\n\n"
            f"🧠 Fidelity = agreement between model's original prediction and its prediction\n"
            f"on perturbed samples within the anchor rule region.\n"
            f"Precision = accuracy of the anchor rule on those samples.\n"
            f"Coverage = proportion of data space the rule applies to.\n"
        )
    else:
        message = "❌ No valid anchor explanations could be computed."

    print(message)
    f.write(message)
