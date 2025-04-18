import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from alibi.explainers import AnchorTabular
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Output file
output_path = "/home/gummadi/spring25/week12/anchor/mems/mems_anchor_fidelity_dnn.txt"

# Load dataset
df = pd.read_csv("/home/gummadi/mems_dataset.csv")
X = df[['x', 'y', 'z']].values
y_raw = df['label'].values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
y_cat = to_categorical(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, stratify=y, test_size=0.2, random_state=42)

# Define simple DNN model
model = Sequential()
model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_cat.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0, validation_split=0.1)

# Define prediction function for Anchor
def predict_fn(x):
    return np.argmax(model.predict(x, verbose=0), axis=1)

# Anchor explainer
feature_names = ['x', 'y', 'z']
explainer = AnchorTabular(predict_fn, feature_names=feature_names)
explainer.fit(X_train, disc_perc=(5, 10, 25, 50, 75, 90, 95))

# Metric collection
fidelity_scores = []
precision_scores = []
coverage_scores = []

for idx in range(min(1000, len(X_test))):
    instance = X_test[idx].reshape(1, -1)
    try:
        explanation = explainer.explain(instance[0], threshold=0.85, n_covered_ex=20000)
        original_pred = predict_fn(instance)[0]
        perturbed_samples = explanation.raw['instances']
        if len(perturbed_samples) == 0:
            continue

        pred_matches = predict_fn(perturbed_samples) == original_pred
        fidelity = np.mean(pred_matches)

        fidelity_scores.append(fidelity)
        precision_scores.append(explanation.precision)
        coverage_scores.append(explanation.coverage)

    except Exception:
        continue

# Write results
with open(output_path, "a") as f:
    if fidelity_scores:
        overall_fidelity = round(np.mean(fidelity_scores), 4)
        avg_precision = round(np.mean(precision_scores), 4)
        avg_coverage = round(np.mean(coverage_scores), 4)

        message = (
            f"✅ Anchor Explanation Summary for Deep Neural Network (Keras)\n"
            f"Instances explained: {len(fidelity_scores)}\n\n"
            f"✅ Overall Fidelity: {overall_fidelity}\n"
            f"✅ Average Precision: {avg_precision}\n"
            f"✅ Average Coverage: {avg_coverage}\n\n"
            f"🧠 Fidelity = agreement between model's original prediction and its prediction\n"
            f"on perturbed samples within the anchor rule region.\n"
            f"Precision = accuracy of the anchor rule on those samples.\n"
            f"Coverage = proportion of data space the rule applies to.\n"
        )
    else:
        message = "❌ No valid anchor explanations could be computed."

    print(message)
    f.write(message)