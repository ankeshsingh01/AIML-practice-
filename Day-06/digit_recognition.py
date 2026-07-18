"""
Day 6 - Handwritten Digit Recognition (Neural Network)
----------------------------------------------------------
First project using a REAL dataset instead of made-up data, and first
neural network in this series.

Dataset: sklearn's built-in "Digits" dataset - 1,797 real 8x8 pixel images
of handwritten digits (0-9), originally from the UCI ML repository. This is
actual scanned handwriting from real people, not synthetic data.
Source: https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset

Concepts covered:
- Working with image data (each image is just a grid of pixel brightness numbers)
- Neural networks (MLPClassifier - Multi-Layer Perceptron)
- Confusion matrix (which digits get confused with which other digits)
- Visualizing predictions against actual images
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# ---------------------------
# 1. Load the REAL digits dataset
# ---------------------------
digits = load_digits()
X = digits.data      # each row = 64 numbers (8x8 pixel grid flattened), representing pixel brightness
y = digits.target    # the actual digit (0-9) each image represents

print(f"Dataset size: {X.shape[0]} images")
print(f"Each image is {digits.images[0].shape} pixels, flattened into {X.shape[1]} numbers")
print(f"Digit classes: {sorted(set(y))}")

# ---------------------------
# 2. Look at a few actual images before doing anything else
# ---------------------------
fig, axes = plt.subplots(1, 8, figsize=(12, 2))
for i, ax in enumerate(axes):
    ax.imshow(digits.images[i], cmap="gray")
    ax.set_title(f"Label: {digits.target[i]}")
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_digits.png")
print("Saved sample_digits.png")

# ---------------------------
# 3. Train/test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# 4. Scale pixel values (neural networks train much better on scaled input)
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# 5. Train a Neural Network (Multi-Layer Perceptron)
# ---------------------------
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),  # two hidden layers: 64 neurons, then 32 neurons
    max_iter=500,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# ---------------------------
# 6. Evaluate
# ---------------------------
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)

print(f"\nTest set accuracy: {round(accuracy * 100, 2)}%")
print("\nClassification report:\n", classification_report(y_test, predictions))

# ---------------------------
# 7. Confusion matrix - which digits does it mix up?
# ---------------------------
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
fig, ax = plt.subplots(figsize=(7, 7))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix - which digits get mixed up?")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Saved confusion_matrix.png")

# ---------------------------
# 8. Show some actual predictions next to the real images
# ---------------------------
fig, axes = plt.subplots(2, 8, figsize=(14, 4))
test_images = X_test.reshape(-1, 8, 8)

for i, ax in enumerate(axes.flat):
    ax.imshow(test_images[i], cmap="gray")
    correct = predictions[i] == y_test[i]
    color = "green" if correct else "red"
    ax.set_title(f"Pred: {predictions[i]}\nActual: {y_test[i]}", color=color, fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("predictions_sample.png")
print("Saved predictions_sample.png")
