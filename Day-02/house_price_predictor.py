"""
Day 3 - House Price Predictor


Given details about a house (area, bedrooms, bathrooms, age, location score),
predict its price using Linear Regression.

Concepts covered:
- Regression vs classification (predicting a number, not a category)
- Feature scaling
- Linear Regression
- Evaluating regression models with R^2 score and Mean Absolute Error
- Comparing against a Random Forest Regressor as a second model
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------
# 1. Small built-in dataset
# Columns: area (sqft), bedrooms, bathrooms, age (years), location_score (1-10)
# Target: price (in lakhs INR)
# ---------------------------
X = np.array([
    [1200, 2, 2, 5,  7],
    [1500, 3, 2, 3,  8],
    [900,  2, 1, 10, 5],
    [2000, 4, 3, 2,  9],
    [1100, 2, 1, 8,  6],
    [1800, 3, 3, 4,  8],
    [750,  1, 1, 15, 4],
    [1600, 3, 2, 6,  7],
    [2500, 4, 4, 1,  10],
    [1000, 2, 1, 12, 5],
    [1350, 3, 2, 7,  6],
    [2200, 4, 3, 3,  9],
    [850,  2, 1, 9,  5],
    [1700, 3, 2, 5,  7],
    [1950, 4, 3, 2,  8],
    [1050, 2, 1, 11, 5],
    [1450, 3, 2, 6,  7],
    [2100, 4, 3, 3,  9],
    [800,  1, 1, 14, 4],
    [1250, 2, 2, 6,  6],
])

y = np.array([
    55, 72, 38, 105, 48, 90, 28, 78, 145, 42,
    62, 118, 36, 82, 100, 44, 70, 112, 30, 58
])

feature_names = ["area", "bedrooms", "bathrooms", "age", "location_score"]

# ---------------------------
# 2. Train/test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ---------------------------
# 3. Scale features (important for Linear Regression when features have very different ranges,
# like area being in the thousands and bedrooms being single digits)
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# 4. Train Linear Regression
# ---------------------------
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_predictions = lr_model.predict(X_test_scaled)

# ---------------------------
# 5. Train Random Forest Regressor for comparison (doesn't need scaling)
# ---------------------------
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# ---------------------------
# 6. Evaluate both models
# ---------------------------
print("=== Linear Regression ===")
print("Actual prices:   ", y_test)
print("Predicted prices:", np.round(lr_predictions, 1))
print("R^2 score:", round(r2_score(y_test, lr_predictions), 3))
print("Mean Absolute Error:", round(mean_absolute_error(y_test, lr_predictions), 2), "lakhs")

print("\n=== Random Forest Regressor ===")
print("Actual prices:   ", y_test)
print("Predicted prices:", np.round(rf_predictions, 1))
print("R^2 score:", round(r2_score(y_test, rf_predictions), 3))
print("Mean Absolute Error:", round(mean_absolute_error(y_test, rf_predictions), 2), "lakhs")

# ---------------------------
# 7. See which features mattered most (Random Forest gives this for free)
# ---------------------------
print("\n=== Feature Importance (Random Forest) ===")
for name, importance in zip(feature_names, rf_model.feature_importances_):
    print(f"{name}: {round(importance, 3)}")

# ---------------------------
# 8. Predict price for a brand new house
# ---------------------------
new_house = np.array([[1400, 3, 2, 5, 7]])  # 1400 sqft, 3 bed, 2 bath, 5 yrs old, location score 7
new_house_scaled = scaler.transform(new_house)

lr_new_pred = lr_model.predict(new_house_scaled)[0]
rf_new_pred = rf_model.predict(new_house)[0]

print("\n=== Predicting a brand new house ===")
print(f"House: 1400 sqft, 3 bed, 2 bath, 5 years old, location score 7")
print(f"Linear Regression predicts: {round(lr_new_pred, 1)} lakhs")
print(f"Random Forest predicts:     {round(rf_new_pred, 1)} lakhs")
