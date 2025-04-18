import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from alibi.explainers import AnchorTabular
from tqdm import tqdm

# Load Dataset
file_path = "/home/gummadi/device7_top_20_features.csv"
df = pd.read_csv(file_path)

# Ensure Equal Samples for Each Class
min_samples = df['label'].value_counts().min()
df_balanced = df.groupby('label').apply(lambda x: x.sample(min_samples, random_state=42)).reset_index(drop=True)

# Feature and Target Separation
X = df_balanced.drop(columns=['label'])  # 20 features
y = df_balanced['label'].values  # 6-class labels

# One-hot encode the target
y = tf.keras.utils.to_categorical(y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build DNN Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model with Progress Bar
model.fit(X_train_scaled, y_train, epochs=50, batch_size=10, verbose=1)

# Model Evaluation
y_pred_probs = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
accuracy = np.mean(y_pred == y_true)
print(f'Model Accuracy: {accuracy:.2f}')

# Anchor Explanation
explainer = AnchorTabular(model.predict, feature_names=X.columns.tolist())
explainer.fit(X_train_scaled, disc_perc=(10, 25, 50, 75, 90))

# Explain Predictions for Each Class
print("Generating explanations...")
for class_label in tqdm(range(y_test.shape[1]), desc="Explaining Class"):
    class_instances = X_test_scaled[np.argmax(y_test, axis=1) == class_label]
    if len(class_instances) == 0:
        print(f"No samples available for class {class_label}")
        continue
    instance_to_explain = class_instances[0]
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
    plt.barh(rules, [precision]*len(rules), align='center', color='skyblue')
    plt.xlabel('Precision')
    plt.ylabel('Rules')
    plt.title(f'Precision of Anchor Rules - Class {class_label}')
    plt.tight_layout()
    
    # Save Graph as JPG
    graph_path = f'/home/gummadi/spring25/week6/d7_anchor_2m_dnn/anchor_explanation_d7_class_{class_label}.jpg'
    plt.savefig(graph_path)
    plt.show()
    
    # Save rules to a file
    with open(f'/home/gummadi/spring25/week6/d7_anchor_2m_dnn/anchor_rules_d7_class_{class_label}.txt', 'w') as f:
        f.write("Anchor Rules:\n")
        f.write("\n".join(rules))
        f.write(f"\n\nPrecision: {precision:.2f}")
        f.write(f"\n\nCoverage: {explanation.coverage:.2f}")
