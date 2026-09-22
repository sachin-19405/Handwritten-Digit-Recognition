"""
Train a Neural Network (MLPClassifier) on the MNIST-style Train.csv
and save the trained model + scaler for deployment.
"""
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time

DATA_PATH = "Train.csv"
MODEL_OUT = "model/digit_model.pkl"
SCALER_OUT = "model/scaler.pkl"

print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Data shape: {df.shape}")

y = df["label"].values
X = df.drop("label", axis=1).values.astype(np.float32)

# Normalize pixel values to 0-1
X = X / 255.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Neural Network: Multi-Layer Perceptron
clf = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=256,
    learning_rate_init=0.001,
    max_iter=40,
    early_stopping=True,
    n_iter_no_change=5,
    validation_fraction=0.1,
    random_state=42,
    verbose=True,
)

print("Training neural network...")
t0 = time.time()
clf.fit(X_train, y_train)
print(f"Training completed in {time.time()-t0:.1f}s")

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred))

joblib.dump(clf, MODEL_OUT)
print(f"Model saved to {MODEL_OUT}")

# Save simple metadata (min/max not needed since scaled by /255 in app too)
joblib.dump({"input_dim": X.shape[1], "classes": list(clf.classes_)}, SCALER_OUT)
print(f"Metadata saved to {SCALER_OUT}")
