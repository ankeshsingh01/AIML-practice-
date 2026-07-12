# Day 3 - House Price Predictor

Switched gears today from classification to regression - basically instead of predicting a category (spam/not spam like Day 1), this model predicts an actual number: the price of a house.

I gave it 20 houses with area, bedrooms, bathrooms, age, and a location score, and it learns to predict price from those. Tried two models side by side to compare - Linear Regression and Random Forest Regressor.

One thing I learned here that I hadn't run into before: since area is in the thousands (like 1200, 1800) but bedrooms is just 2-4, Linear Regression gets confused if you don't scale the features first. So I used StandardScaler to bring everything onto a similar range before training. Random Forest doesn't actually need this since it splits on thresholds rather than distances, so I trained it on the raw unscaled data instead.

Surprising result: Linear Regression actually beat Random Forest here (R² of 0.98 vs 0.76). Usually people assume Random Forest = better because it's "fancier," but with only 20 data points it doesn't have enough examples to build good decision trees, while a straight line fit the relationship between size/location and price pretty well. Another good lesson in "bigger model isn't always better with small data."

Also pulled out feature importance from the Random Forest model - turns out area and location score mattered most for price, which matches common sense.

### Stack
Python, Scikit-learn, Linear Regression, Random Forest Regressor, StandardScaler

### Running it
```bash
pip install scikit-learn numpy
python house_price_predictor.py
```

It trains both models, prints R² score and Mean Absolute Error for each, shows which features mattered most, then predicts the price of a brand new house I made up.

### What I'd do next
- Get a real housing dataset (like the Boston or California housing dataset) with way more rows and see if Random Forest starts winning once there's more data
- Try Ridge/Lasso regression to see if regularization helps
- Plot actual vs predicted prices to visualize how close the model gets
