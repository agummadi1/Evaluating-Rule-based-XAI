import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance
from anchor import anchor_tabular

# Start Execution Timer
start_time = time.time()

# Combine feature columns with the label column
req_cols = ['y', 'label']
num_columns = 3  # 3 features (x, y, z)
num_labels = 3  # 3-class classification

split = 0.70  # Train/Test split ratio

# Model Parameters
kernel = 'rbf'  # Kernel type for SVM
C = 1.0  # Regularization parameter

# Ensure Exactly `1000` Samples
samples = 1000

# Output file
output_file_name = "/home/gummadi/spring25/week5/mems_ANCHOR_4m_svm_output.txt"

print('--------------------------------------------------')
print('SVM with ANCHOR Explainability')
print('--------------------------------------------------')

# Load dataset
df = pd.read_csv('/home/gummadi/mems_dataset.csv', usecols=req_cols)

# Ensure Exactly `1000` Samples
if len(df) > samples:
    df = df.sample(n=samples, random_state=42)
elif len(df) < samples:
    df = df.sample(n=samples, replace=True, random_state=42)  # Upsampling if needed

# Separate features and labels
X = df.drop(columns=['label'])
y = df['label']

# Map labels from 1 to 3 to 0 to 2
y -= 1

# Standardizing numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=split, random_state=42)

# Define SVM Model
model = SVC(kernel=kernel, C=C, probability=True, random_state=42)

# Train Model
train_start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - train_start_time

# Evaluate Model
pred_start_time = time.time()
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
prediction_time = time.time() - pred_start_time

# Classification Report
class_report = classification_report(y_test, y_pred)

# Compute Feature Importance Using Permutation Importance
perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
feature_importances = perm_importance.importances_mean

# Sort Feature Importances
sorted_features = sorted(zip(['y'], feature_importances), key=lambda x: x[1], reverse=True)

# Sparsity Values Calculation
thresholds = np.linspace(0, 1, 10)
sparsity_values = [sum(abs(feature_importances) < t) / len(feature_importances) for t in thresholds]

# Anchor Explainer
explainer = anchor_tabular.AnchorTabularExplainer(
    class_names=["Normal", "Near-failure", "Failure"],  # Map labels to class names
    feature_names=['y'],
    train_data=X_train
)

# Explain one test sample with Anchor
exp = explainer.explain_instance(X_test[0], model.predict, threshold=0.99)

# Extracting Rule-based Explanation
anchor_rules = exp.names()
precision = float(exp.exp_map['precision'][0]) if isinstance(exp.exp_map['precision'], list) else float(exp.exp_map['precision'])
coverage = float(exp.exp_map['coverage'][0]) if isinstance(exp.exp_map['coverage'], list) else float(exp.exp_map['coverage'])

# Write results to output file
with open(output_file_name, "a") as f:
    f.write("\n")
    f.write("------------------------------------------------------------------------------------------\n")
    f.write("### MEMS Dataset Model Evaluation ###\n")
    f.write(f"Total Samples Used: {samples}\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Testing Samples: {len(X_test)}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Training Time: {training_time:.4f} seconds\n")
    f.write(f"Prediction Time: {prediction_time:.4f} seconds\n")
    f.write(f"Full Execution Time: {time.time() - start_time:.4f} seconds\n\n")
    
    f.write("### Feature Importances (Descending Order) ###\n")
    for feature, importance in sorted_features:
        f.write(f"{feature}: {importance:.4f}\n")
    f.write("\n")
    
    f.write("### Sparsity Values ###\n")
    for val in sparsity_values:
        f.write(f"{val}\n")
    
    f.write("\n### Anchor Explanation for First Test Sample ###\n")
    f.write(f"Prediction: {exp.exp_map['prediction']}\n")
    f.write("Anchor Rules:\n")
    f.write("\n".join(anchor_rules) + "\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Coverage: {coverage:.4f}\n\n")
    f.write(f"Execution time: {(time.time() - start_time) / 3600:.6f} hours\n")

# Print Execution Time to Console
print("Execution time: %s hours" % ((time.time() - start_time) / 3600))

# Confirm Results Saved
print(f"Results saved to {output_file_name}")
