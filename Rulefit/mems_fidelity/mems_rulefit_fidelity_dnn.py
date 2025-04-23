import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from rulefit import RuleFit

# Output file
output_path = "/home/gummadi/spring25/week13/rulefit/mems/mems_rulefit_fidelity_dnn.txt"

# Load dataset
df = pd.read_csv("/home/gummadi/mems_dataset.csv")
X = df[['x', 'y', 'z']]
y = LabelEncoder().fit_transform(df['label'].values)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Build DNN model
dnn_model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])
dnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train DNN
dnn_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

# Predict on train and test
dnn_preds_train_raw = dnn_model.predict(X_train).flatten()
dnn_preds_test_raw = dnn_model.predict(X_test).flatten()
dnn_preds_train = (dnn_preds_train_raw >= 0.5).astype(int)
dnn_preds_test = (dnn_preds_test_raw >= 0.5).astype(int)

# Train RuleFit
rulefit_model = RuleFit(
    tree_generator=GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=42),
    max_rules=500,
    memory_par=0.01
)
rulefit_model.include_linear = False

# Fit RuleFit on DNN's *train* predictions
rulefit_model.fit(X_train.values, dnn_preds_train, feature_names=X.columns)

# Predict with RuleFit
rulefit_preds_raw = rulefit_model.predict(X_test.values)
rulefit_preds = (rulefit_preds_raw >= 0.5).astype(int)

# Fidelity: Agreement between RuleFit and DNN
fidelity = accuracy_score(dnn_preds_test, rulefit_preds)

# Write output to file
with open(output_path, 'w') as f:
    f.write(f"✅ RuleFit Fidelity to DNN: {fidelity:.4f}\n")

print("Fidelity written to file:", output_path)
