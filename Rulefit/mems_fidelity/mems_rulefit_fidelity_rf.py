import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from rulefit import RuleFit
from sklearn.metrics import accuracy_score

# Output file
output_path = "/home/gummadi/mems_rulefit_fidelity_rf.txt"

# Load dataset
df = pd.read_csv("/home/gummadi/mems_dataset.csv")
X = df[['x', 'y', 'z']]
y = LabelEncoder().fit_transform(df['label'].values)

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
# Important: set include_linear separately
rulefit_model.include_linear = False

# Fit RuleFit
rulefit_model.fit(X_train.values, rf_preds_train, feature_names=X.columns)

# Predict with RuleFit
rulefit_preds_raw = rulefit_model.predict(X_test.values)
rulefit_preds = (rulefit_preds_raw >= 0.5).astype(int)

# Fidelity: Agreement between RuleFit and RF
fidelity = accuracy_score(rf_preds_test, rulefit_preds)

# Write output to file
with open(output_path, 'w') as f:
    f.write(f"✅ Improved RuleFit Fidelity to Random Forest: {fidelity:.4f}\n")

print("Improved Fidelity written to file:", output_path)
