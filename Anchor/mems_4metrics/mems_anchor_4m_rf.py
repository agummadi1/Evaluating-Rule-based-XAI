import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from anchor import anchor_tabular

# Start Execution Timer
start_time = time.time()

# Combine feature columns with the label column
req_cols = ['x', 'y', 'z', 'label']
num_columns = 3  # 3 features (x, y, z)
num_labels = 3  # 3-class classification

split = 0.70  # Train/Test split ratio

# Model Parameters
n_estimators = 100  # Number of trees in the forest
max_depth = None  # Maximum depth of the tree

# Ensure Exactly `1000` Samples
samples = 1000

# Output file
output_file_name = "mems_ANCHOR_4m_rf_output.txt"
sparsity_output_file = "mems_ANCHOR_rf_sparsity_values.txt"
plot_output_file = "mems_ANCHOR_rf_sparsity_plot.jpg"

print('--------------------------------------------------')
print('Random Forest with ANCHOR Explainability')
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

# Define Random Forest Model
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

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

# Feature Importance
feature_importances = model.feature_importances_
sorted_features = sorted(zip(['x', 'y', 'z'], feature_importances), key=lambda x: x[1], reverse=True)

# Anchor Explainer
explainer = anchor_tabular.AnchorTabularExplainer(
    class_names=["Normal", "Near-failure", "Failure"],  # Map labels to class names
    feature_names=['x', 'y', 'z'],
    train_data=X_train
)

# Explain one test sample with Anchor
exp = explainer.explain_instance(X_test[0], model.predict, threshold=0.99)

# Extracting Rule-based Explanation
anchor_rules = exp.names()
precision = float(exp.exp_map['precision'][0]) if isinstance(exp.exp_map['precision'], list) else float(exp.exp_map['precision'])
coverage = float(exp.exp_map['coverage'][0]) if isinstance(exp.exp_map['coverage'], list) else float(exp.exp_map['coverage'])

# Sparsity Values Calculation
thresholds = np.linspace(0, 1, 10)
sparsity_values = [sum(abs(feature_importances) < t) / len(feature_importances) for t in thresholds]

# Save Sparsity Values to File
with open(sparsity_output_file, "w") as f:
    for val in sparsity_values:
        f.write(f"{val}\n")

# Plot Sparsity Graph
plt.figure()
plt.plot(thresholds, sparsity_values, 'bo--', label='Random Forest Feature Sparsity')
plt.xlabel('Threshold')
plt.ylabel('Sparsity')
plt.legend()
plt.grid(True)
plt.savefig(plot_output_file)
plt.close()

# End Execution Timer
end_time = time.time()
full_execution_time = end_time - start_time
execution_time_hours = full_execution_time / 3600  # Convert to hours

# Write results to output file
with open(output_file_name, "a") as f:
    f.write("------------------------------------------------------------------------------------------\n")
    f.write("### MEMS Dataset Model Evaluation ###\n")
    f.write(f"Total Samples Used: {samples}\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Testing Samples: {len(X_test)}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Training Time: {training_time:.4f} seconds\n")
    f.write(f"Prediction Time: {prediction_time:.4f} seconds\n")
    f.write(f"Full Execution Time: {full_execution_time:.4f} seconds\n\n")
    
    f.write("### Feature Importances (Descending Order) ###\n")
    for feature, importance in sorted_features:
        f.write(f"{feature}: {importance:.4f}\n")
    f.write("\n")

    f.write("### Anchor Explanation for First Test Sample ###\n")
    f.write(f"Prediction: {exp.exp_map['prediction']}\n")
    f.write("Anchor Rules:\n")
    f.write("\n".join(anchor_rules) + "\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Coverage: {coverage:.4f}\n")
    f.write(f"Execution time: {execution_time_hours:.6f} hours\n")

# Print Execution Time to Console
print("Execution time: %s hours" % execution_time_hours)

# Confirm Results Saved
print(f"Results saved to {output_file_name}")
print(f"Sparsity values saved to {sparsity_output_file}")
print(f"Plot saved as {plot_output_file}")
