import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from alibi.explainers import AnchorTabular
from tqdm import tqdm

# Load Dataset
file_path = "/home/gummadi/device7_top_20_features.csv"
df = pd.read_csv(file_path)

# Sample for quicker iterations
df_sampled = df.sample(frac=0.5, random_state=42)

# Feature and Target Separation
X = df_sampled.drop(columns=['label'])
y = df_sampled['label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Classifier Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42))
])

# Train Model
print("Training model...")
pipeline.fit(X_train, y_train)

# Model Evaluation
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Write Model Accuracy to File
# with open('/home/gummadi/spring25/week6/d7_anchor_2m_svm/svm_anchor_model_accuracy.txt', 'w') as f:
#     f.write(f'Model Accuracy: {accuracy:.2f}\n')

# Anchor Explanation
explainer = AnchorTabular(pipeline.predict, feature_names=X.columns.tolist())
explainer.fit(X_train.values, disc_perc=(25, 50, 75))

# Explain Predictions and Write to File
print("Generating explanations...")
for class_label in tqdm(sorted(y.unique()), desc="Explaining Class"):
    file_path = f'/home/gummadi/spring25/week6/d7_anchor_2m_svm/svm_anchor_explanation_class_{class_label}.txt'
    with open(file_path, 'w') as f:
        f.write(f"Explaining Predictions for Class {class_label}\n")
        class_instances = X_test[y_test == class_label]
        if class_instances.empty:
            f.write(f"No samples available for class {class_label}\n")
            continue
        instance_to_explain = class_instances.iloc[0].values
        explanation = explainer.explain(instance_to_explain, threshold=0.90)
        # Write Rules
        f.write("Anchor Rules:\n")
        f.write("\n".join(explanation.anchor))
        f.write(f"\nPrecision: {explanation.precision:.2f}\n")
        f.write(f"Coverage: {explanation.coverage:.2f}\n")
        
        # Graphical Representation of Rules
        plt.figure(figsize=(10, 6))
        rules = explanation.anchor
        precision = explanation.precision
        plt.barh(rules, [precision]*len(rules), align='center', color='skyblue')
        plt.xlabel('Precision')
        plt.ylabel('Rules')
        plt.title(f'Precision of Anchor Rules with SVM - Class {class_label}')
        plt.tight_layout()
        
        # Save Graph as JPG
        graph_path = f'/home/gummadi/spring25/week6/d7_anchor_2m_svm/svm_anchor_explanation_d7_class_{class_label}.jpg'
        plt.savefig(graph_path)
        plt.close()  # Close the figure to avoid display
