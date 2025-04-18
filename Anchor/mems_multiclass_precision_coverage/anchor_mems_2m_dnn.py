import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from alibi.explainers import AnchorTabular

# Load Dataset
df = pd.read_csv('/home/gummadi/mems_dataset.csv')  # Adjust file path as needed

# Ensure Equal Samples for Each Class
min_samples = df['label'].value_counts().min()
df_balanced = df.groupby('label').apply(lambda x: x.sample(min_samples, random_state=42)).reset_index(drop=True)

# Feature and Target Separation
X = df_balanced[['x', 'y', 'z']]
y = df_balanced['label']  # Assuming labels are integers: 1 = Normal, 2 = Near-failure, 3 = Failure

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert Labels to Categorical
y_train_cat = keras.utils.to_categorical(y_train - 1, num_classes=3)
y_test_cat = keras.utils.to_categorical(y_test - 1, num_classes=3)

# Build DNN Model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(X_train_scaled, y_train_cat, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test_cat))

# Model Evaluation
loss, accuracy = model.evaluate(X_test_scaled, y_test_cat)
print(f'Model Accuracy: {accuracy:.2f}')

# Anchor Explanation
class_names = ['Normal', 'Near-failure', 'Failure']
def model_predict(X):
    return np.argmax(model.predict(scaler.transform(X)), axis=1)

explainer = AnchorTabular(model_predict, feature_names=['x', 'y', 'z'])
explainer.fit(X_train.values, disc_perc=(10, 25, 50, 75, 90))  # Enhanced Discretization

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
    graph_path = f'/home/gummadi/spring25/week6/anchor_mems_2m_dnn/anchor_mems_DNN_explanation_class_{class_label}.jpg'
    plt.savefig(graph_path)
    plt.show()
    
    # Save rules to a file
    with open(f'/home/gummadi/spring25/week6/anchor_mems_2m_dnn/anchor_mems_DNN_rules_class_{class_label}.txt', 'w') as f:
        f.write("Anchor Rules:\n")
        f.write("\n".join(rules))
        f.write(f"\n\nPrecision: {precision:.2f}")
        f.write(f"\n\nCoverage: {explanation.coverage:.2f}")
