import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from rulefit import RuleFit

# Output file
output_path = "/home/gummadi/spring25/week13/rulefit/d7/d7_rulefit_fidelity_rf.txt"

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

# Train original Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on train and test
rf_preds_train = rf_model.predict(X_train)
rf_preds_test = rf_model.predict(X_test)

# Train RuleFit
rulefit_model = RuleFit(
    tree_generator=GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=42),
    max_rules=500,
    memory_par=0.01
)
rulefit_model.include_linear = False

# Fit RuleFit on Random Forest's *train* predictions
rulefit_model.fit(X_train, rf_preds_train, feature_names=feature_names)

# Predict with RuleFit
rulefit_preds_raw = rulefit_model.predict(X_test)
rulefit_preds = (rulefit_preds_raw >= 0.5).astype(int)

# Fidelity: Agreement between RuleFit and Random Forest
fidelity = accuracy_score(rf_preds_test, rulefit_preds)

# Write output to file
with open(output_path, 'w') as f:
    f.write(f"✅ RuleFit Fidelity to Random Forest (device7): {fidelity:.4f}\n")

print("Fidelity written to file:", output_path)
