# Day 8 - Breast Cancer Classification

Real medical dataset today - the Breast Cancer Wisconsin (Diagnostic) dataset, built into scikit-learn. 569 real patient tumor samples, each with 30 measurements taken from digitized biopsy images (radius, texture, smoothness, concavity, etc.), labeled with the actual diagnosis: malignant or benign.

Source: https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset (originally UCI ML Repository)

## Why this dataset is a good exercise in more than just "get high accuracy"

This is the first project where accuracy alone isn't the most important number. In a medical diagnosis context, **a false negative (predicting benign when it's actually malignant) is far more dangerous than a false positive** - a false positive just means an extra test gets ordered; a false negative means a real cancer gets missed entirely. So recall on the malignant class matters more than overall accuracy.

## Results

- Accuracy: **98.25%**
- Precision: **98.61%**
- Recall: **98.61%**
- F1 Score: **0.986**

Looked specifically at the confusion matrix instead of just trusting the headline accuracy number: out of 114 test patients, there was **1 false negative** - one malignant case predicted as benign. In a real deployed system, this is exactly the number you'd want to drive down as close to zero as possible, even if it costs some accuracy elsewhere (more false positives are an acceptable tradeoff for fewer missed cancers).

## Feature importance

Pulled out which of the 30 measurements mattered most to the model (`images/feature_importance.png`). Features related to cell size/area and "worst" (most extreme) measurements across the cells in a sample tended to matter most - which lines up with medical intuition, since more irregular and larger cell structures are a classic indicator of malignancy.

## Stack
Python, Scikit-learn (LogisticRegression, StandardScaler), Matplotlib, NumPy

## Running it
```bash
pip install scikit-learn matplotlib numpy
python breast_cancer_classifier.py
```

## What I'd try next
- Try to push recall on the malignant class even higher by adjusting the classification threshold (trading some precision for it)
- Compare against Random Forest or SVM and see if either reduces false negatives further
- Look at which specific patient was misclassified and check if their measurements were borderline/ambiguous
