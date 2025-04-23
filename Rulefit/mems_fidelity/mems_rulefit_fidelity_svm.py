import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from rulefit import RuleFit
from sklearn.metrics import accuracy_score

# Output file
output_path = "/home/gummadi/spring25/week13/rulefit/mems/mems_rulefit_fidelity_svm.txt"

# Load dataset
df = pd.read_csv("/home/gummadi/mems_dataset.csv")
X = df[['x', 'y', 'z']]
y = LabelEncoder().fit_transform(df['label'].values)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train original SVM
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Predict on train and test
svm_preds_train = svm_model.predict(X_train)
svm_preds_test = svm_model.predict(X_test)

# Train RuleFit
rulefit_model = RuleFit(
    tree_generator=GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=42),
    max_rules=500,
    memory_par=0.01
)
# Important: set include_linear separately
rulefit_model.include_linear = False

# Fit RuleFit on SVM's *train* predictions
rulefit_model.fit(X_train.values, svm_preds_train, feature_names=X.columns)

# Predict with RuleFit
rulefit_preds_raw = rulefit_model.predict(X_test.values)
rulefit_preds = (rulefit_preds_raw >= 0.5).astype(int)

# Fidelity: Agreement between RuleFit and SVM
fidelity = accuracy_score(svm_preds_test, rulefit_preds)

# Write output to file
with open(output_path, 'w') as f:
    f.write(f"✅ RuleFit Fidelity to SVM: {fidelity:.4f}\n")

print("Fidelity written to file:", output_path)
