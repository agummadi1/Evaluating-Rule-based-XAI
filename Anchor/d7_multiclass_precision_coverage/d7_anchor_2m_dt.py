import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from alibi.explainers import AnchorTabular

# Load Dataset
file_path = "/home/gummadi/device7_top_20_features.csv"
df = pd.read_csv(file_path)

# Ensure Equal Samples for Each Class
min_samples = df['label'].value_counts().min()
df_balanced = df.groupby('label').apply(lambda x: x.sample(min_samples, random_state=42)).reset_index(drop=True)

# Feature and Target Separation
X = df_balanced.drop(columns=['label'])  # 20 features
y = df_balanced['label']  # 6-class labels

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Classifier Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', DecisionTreeClassifier(random_state=42, max_depth=10))
])

# Train Model
pipeline.fit(X_train, y_train)

# Model Evaluation
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy:.2f}')

# Anchor Explanation
class_names = ['Benign', 'gafgyt.combo', 'gafgyt.junk', 'gafgyt.scan', 'gafgyt.tcp', 'gafgyt.udp']
explainer = AnchorTabular(pipeline.predict, feature_names=X.columns.tolist())
explainer.fit(X_train.values, disc_perc=(10, 25, 50, 75, 90))  # Enhanced Discretization of features

# Explain Predictions for Each Class
for class_label in sorted(y.unique()):
    print(f"\nExplaining Predictions for Class {class_label}")
    class_instances = X_test[y_test == class_label]
    if class_instances.empty:
        print(f"No samples available for class {class_label}")
        continue
    instance_to_explain = class_instances.iloc[0].values
    explanation = explainer.explain(instance_to_explain, threshold=0.90)
    
    # Print Rules
    print("Anchor Rules:")
    print("\n".join(explanation.anchor))
    print(f"Precision: {explanation.precision:.2f}")
    print(f"Coverage: {explanation.coverage:.2f}")

    # Graphical Representation of Rules
    plt.figure(figsize=(10, 6))
    rules = explanation.anchor
    precision = explanation.precision
    plt.barh(rules, precision, align='center', color='skyblue')
    plt.xlabel('Precision')
    plt.ylabel('Rules')
    plt.title(f'Precision of Anchor Rules - Class {class_label}')
    plt.tight_layout()
    
    # Save Graph as JPG
    graph_path = f'/home/gummadi/spring25/week6/d7_anchor_2m_dt/anchor_explanation_d7_class_{class_label}.jpg'
    plt.savefig(graph_path)
    plt.show()
    
    # Save rules to a file
    with open(f'/home/gummadi/spring25/week6/d7_anchor_2m_dt/anchor_rules_d7_class_{class_label}.txt', 'w') as f:
        f.write("Anchor Rules:\n")
        f.write("\n".join(rules))
        f.write(f"\n\nPrecision: {precision:.2f}")
        f.write(f"\n\nCoverage: {explanation.coverage:.2f}")