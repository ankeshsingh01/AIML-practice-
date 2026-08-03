"""
Day 9 - Hyperparameter Tuning (GridSearchCV vs RandomizedSearchCV)
------------------------------------------------------------------------
Every model so far used mostly default settings. Today's about actually
tuning a model properly - the difference between "I trained a model" and
"I trained a properly optimized model."

Using the same real Breast Cancer Wisconsin dataset from Day 8, but this
time with a Random Forest (which has several hyperparameters worth tuning,
unlike Logistic Regression which has fewer meaningful knobs to turn).

Concepts covered:
- Hyperparameters vs parameters (parameters are LEARNED during training;
  hyperparameters are settings WE choose before training even starts)
- GridSearchCV - exhaustively tries every combination in a defined grid
- RandomizedSearchCV - samples random combinations instead of trying all of
  them (much faster when the grid is large, usually finds a nearly-as-good
  result)
- Cross-validation - each combination is tested with k-fold cross-validation,
  not just one train/test split, so the result is more trustworthy
- Comparing baseline (default settings) vs tuned model on the SAME held-out
  test set, to see if tuning actually helped
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

# ---------------------------
# 1. Load the real dataset (same one from Day 8)
# ---------------------------
data = load_breast_cancer()
X, y = data.data, data.target
print(f"Dataset: {X.shape[0]} real patient samples, {X.shape[1]} features")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# 2. Baseline: Random Forest with all default settings
# ---------------------------
baseline_model = RandomForestClassifier(random_state=42)
baseline_model.fit(X_train, y_train)
baseline_predictions = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_predictions)
baseline_f1 = f1_score(y_test, baseline_predictions)
baseline_recall = recall_score(y_test, baseline_predictions, pos_label=0)  # recall on malignant

print(f"\n=== Baseline (default settings) ===")
print(f"Accuracy: {round(baseline_accuracy*100, 2)}%, F1: {round(baseline_f1, 4)}, Recall (malignant): {round(baseline_recall*100, 2)}%")

# ---------------------------
# 3. GridSearchCV - tries EVERY combination in this grid
# 3 x 3 x 2 x 2 = 36 combinations, each tested with 5-fold CV = 180 model fits
# ---------------------------
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

print(f"\nGridSearchCV: testing {3*3*2*2} combinations x 5-fold CV = {3*3*2*2*5} model fits...")
start_time = time.time()

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
grid_time = time.time() - start_time

grid_predictions = grid_search.predict(X_test)
grid_accuracy = accuracy_score(y_test, grid_predictions)
grid_f1 = f1_score(y_test, grid_predictions)
grid_recall = recall_score(y_test, grid_predictions, pos_label=0)

print(f"GridSearchCV took {round(grid_time, 1)} seconds")
print(f"Best params: {grid_search.best_params_}")
print(f"Tuned - Accuracy: {round(grid_accuracy*100, 2)}%, F1: {round(grid_f1, 4)}, Recall (malignant): {round(grid_recall*100, 2)}%")

# ---------------------------
# 4. RandomizedSearchCV - samples a fixed number of RANDOM combinations
# instead of trying all of them. Bigger search space here to show where
# this approach shines: when the full grid would be too expensive.
# ---------------------------
param_distributions = {
    "n_estimators": [50, 100, 150, 200, 250, 300],
    "max_depth": [None, 3, 5, 7, 10, 15, 20],
    "min_samples_split": [2, 3, 4, 5, 6, 8, 10],
    "min_samples_leaf": [1, 2, 3, 4, 5],
    "max_features": ["sqrt", "log2", None]
}
full_grid_size = 6*7*7*5*3

print(f"\nRandomizedSearchCV: full grid would be {full_grid_size} combos x 5-fold = {full_grid_size*5} fits, sampling only 30 x 5-fold = 150 instead...")
start_time = time.time()

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=30,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
random_time = time.time() - start_time

random_predictions = random_search.predict(X_test)
random_accuracy = accuracy_score(y_test, random_predictions)
random_f1 = f1_score(y_test, random_predictions)
random_recall = recall_score(y_test, random_predictions, pos_label=0)

print(f"RandomizedSearchCV took {round(random_time, 1)} seconds")
print(f"Best params: {random_search.best_params_}")
print(f"Tuned - Accuracy: {round(random_accuracy*100, 2)}%, F1: {round(random_f1, 4)}, Recall (malignant): {round(random_recall*100, 2)}%")

# ---------------------------
# 5. Summary comparison
# ---------------------------
print("\n=== Summary ===")
print(f"{'Model':<22}{'Accuracy':<12}{'F1':<10}{'Recall(malig)':<16}{'Search Time'}")
print(f"{'Baseline':<22}{round(baseline_accuracy*100,2):<12}{round(baseline_f1,4):<10}{round(baseline_recall*100,2):<16}{'0s'}")
print(f"{'GridSearchCV':<22}{round(grid_accuracy*100,2):<12}{round(grid_f1,4):<10}{round(grid_recall*100,2):<16}{round(grid_time,1)}s")
print(f"{'RandomizedSearchCV':<22}{round(random_accuracy*100,2):<12}{round(random_f1,4):<10}{round(random_recall*100,2):<16}{round(random_time,1)}s")

# ---------------------------
# 6. Visualize the comparison
# ---------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

models = ["Baseline\n(defaults)", "GridSearchCV\n(tuned)", "RandomizedSearchCV\n(tuned)"]
f1_scores = [baseline_f1, grid_f1, random_f1]
colors = ["#95a5a6", "#3498db", "#2ecc71"]

ax1.bar(models, f1_scores, color=colors)
ax1.set_ylabel("F1 Score")
ax1.set_title("Model Performance Comparison")
ax1.set_ylim(0.9, 1.0)

search_times = [0, grid_time, random_time]
ax2.bar(models, search_times, color=colors)
ax2.set_ylabel("Search Time (seconds)")
ax2.set_title("Time Cost of Tuning")

plt.tight_layout()
plt.savefig("images/tuning_comparison.png")
print("\nSaved images/tuning_comparison.png")
