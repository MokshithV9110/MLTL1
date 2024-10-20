import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data: [Square footage, Number of bedrooms, Number of bathrooms]
data = {
    'Square footage': [1500, 2000, 2500, 3000, 3500, 4000, 4500],
    'Bedrooms': [3, 4, 3, 5, 4, 4, 5],
    'Bathrooms': [2, 3, 2, 4, 3, 3, 4],
    'Price': [300000, 400000, 350000, 500000, 450000, 600000, 550000]
}

# Convert data into a pandas DataFrame
df = pd.DataFrame(data)

# Features (Square footage, Bedrooms, Bathrooms)
X = df[['Square footage', 'Bedrooms', 'Bathrooms']]

# Target variable (Price)
y = df['Price']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Output the model coefficients and evaluation metrics
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Root Mean Squared Error (RMSE):", rmse)

# Comparing actual vs predicted values
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nActual vs Predicted Prices:")
print(comparison_df)

