"""
Day 8 - Breast Cancer Classification
----------------------------------------
Real medical diagnostic dataset - 569 real patient records from the
Breast Cancer Wisconsin (Diagnostic) dataset, built into scikit-learn.
Each row is a real tumor sample, with 30 measurements taken from a digitized
image of a biopsy (radius, texture, smoothness, etc.), and the real
diagnosis: malignant (cancerous) or benign (not cancerous).

Source: https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset
(originally from UCI Machine Learning Repository)

Concepts covered:
- Working with real medical/high-stakes data
- Logistic Regression for binary classification
- Why recall matters more than accuracy in medical contexts (missing a
  cancer case is far worse than a false alarm)
- Feature importance - which measurements matter most for diagnosis
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

# ---------------------------
# 1. Load the REAL dataset
# ---------------------------
data = load_breast_cancer()
X = data.data
y = data.target  # 0 = malignant, 1 = benign
feature_names = data.feature_names

print(f"Dataset: {X.shape[0]} real patient samples, {X.shape[1]} features each")
print(f"Classes: {dict(zip(data.target_names, [0, 1]))}")
print(f"Class balance: {sum(y==0)} malignant, {sum(y==1)} benign")

# ---------------------------
# 2. Train/test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# 3. Scale features (measurements are on very different scales - e.g. area vs smoothness)
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# 4. Train Logistic Regression
# ---------------------------
model = LogisticRegression(max_iter=5000, random_state=42)
model.fit(X_train_scaled, y_train)

# ---------------------------
# 5. Evaluate
# ---------------------------
predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f"\nAccuracy:  {round(accuracy * 100, 2)}%")
print(f"Precision: {round(precision * 100, 2)}%")
print(f"Recall:    {round(recall * 100, 2)}%")
print(f"F1 Score:  {round(f1, 3)}")
print("\n", classification_report(y_test, predictions, target_names=data.target_names))

# ---------------------------
# 6. Confusion matrix - critically, check how many malignant cases got missed
# ---------------------------
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Reds", colorbar=False)
plt.title("Confusion Matrix - Breast Cancer Diagnosis")
plt.tight_layout()
plt.savefig("images/confusion_matrix.png")
print("Saved images/confusion_matrix.png")

false_negatives = cm[0][1]  # actually malignant, predicted benign - the dangerous error
print(f"\nFalse negatives (missed cancer cases): {false_negatives}")

# ---------------------------
# 7. Which features mattered most?
# Sort by MAGNITUDE (absolute value) so all bars point the same direction -
# color tells us the direction of impact instead (green = pushes toward
# benign, red = pushes toward malignant), which is much easier to read
# than mixing bar direction AND magnitude together.
# ---------------------------
coefficients = model.coef_[0]
top_features_idx = np.argsort(np.abs(coefficients))[-10:]
top_coefs = coefficients[top_features_idx]
top_names = [feature_names[i] for i in top_features_idx]
bar_colors = ["#e74c3c" if c < 0 else "#2ecc71" for c in top_coefs]

plt.figure(figsize=(8, 6))
plt.barh(range(10), np.abs(top_coefs), color=bar_colors)
plt.yticks(range(10), top_names)
plt.xlabel("Impact on prediction (magnitude)")
plt.title("Top 10 Most Important Features\n(green = pushes toward benign, red = pushes toward malignant)")
plt.tight_layout()
plt.savefig("images/feature_importance.png")
print("Saved images/feature_importance.png")
