# Day 9 - Hyperparameter Tuning (GridSearchCV vs RandomizedSearchCV)

Every model up to this point used mostly default settings. Today's about actually tuning a model properly - understanding the difference between "I trained a model" and "I trained a properly optimized model." Used the same real Breast Cancer Wisconsin dataset from Day 8, but with Random Forest instead of Logistic Regression, since Random Forest has several hyperparameters actually worth tuning.

## Parameters vs hyperparameters

Parameters are what the model LEARNS during training (like the coefficients in Day 8's Logistic Regression). Hyperparameters are settings WE choose before training even starts - things like how many trees to build, how deep each tree can go, how many samples are needed before splitting a node. The model can't learn these on its own; we have to search for good values.

## Two search strategies

- **GridSearchCV** - exhaustively tries every single combination in a defined grid. Tested 36 combinations (`n_estimators` x `max_depth` x `min_samples_split` x `min_samples_leaf`), each evaluated with 5-fold cross-validation = 180 total model fits. Took 28.4 seconds.
- **RandomizedSearchCV** - instead of trying everything, randomly samples a fixed number of combinations from a (much bigger) search space. The full grid here would have been 4,410 combinations (22,050 fits with 5-fold CV) - way too expensive to search exhaustively. Sampled just 30 combinations instead (150 fits) and still found a solid result. Took 50.5 seconds (the search space per combination was more complex here, hence longer despite fewer fits).

Both use k-fold cross-validation internally (not just one train/test split) so the "best" combination found is more trustworthy than if it were judged on a single lucky/unlucky split.

## The honest result

| Model | Accuracy | F1 Score | Recall (malignant) | Search Time |
|---|---|---|---|---|
| Baseline (defaults) | 95.61% | 0.9655 | 92.86% | 0s |
| GridSearchCV (tuned) | 94.74% | 0.9583 | 92.86% | 28.4s |
| RandomizedSearchCV (tuned) | 94.74% | 0.9583 | 92.86% | 50.5s |

Tuning actually made the test-set accuracy slightly WORSE here, not better. This surprised me at first, but it's a genuinely important lesson, not a mistake in the code: GridSearchCV picks the best hyperparameters based on cross-validation performance on the TRAINING data, not the held-out test set. With only 569 samples total, there's enough variance between the CV folds and the final test split that "best on CV" doesn't perfectly guarantee "best on this specific test set." The default Random Forest settings happened to generalize slightly better to this particular test split.

This is actually a really common real-world outcome, especially on smaller datasets - tuning helps on average and over many runs, but isn't guaranteed to beat defaults on any single test set. Recall on the malignant class (the metric that matters most medically, per Day 8's reasoning) stayed identical across all three models, which is somewhat reassuring - the tuning didn't make the dangerous-error rate worse, even if overall accuracy dipped slightly.

## Stack
Python, Scikit-learn (RandomForestClassifier, GridSearchCV, RandomizedSearchCV), Matplotlib

## Running it
```bash
pip install scikit-learn matplotlib numpy
python hyperparameter_tuning.py
```

## What I'd try next
- Run this multiple times with different random train/test splits and average the results, to see if tuning wins "on average" even though it lost on this particular split
- Try tuning with a larger dataset where the CV estimate would be more stable
- Try Bayesian optimization (e.g. Optuna) instead of grid/random search, which is generally more sample-efficient
