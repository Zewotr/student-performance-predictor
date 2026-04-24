import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load the dataset
df = pd.read_csv("../student-performance-predictor/data/student_performance.csv")

# Preprocess the data
target_column = "final_score"
X = df.drop(target_column, axis=1)
y = df[target_column]

# Handle missing values
X = X.fillna(X.mean(numeric_only=True))

# Convert categorical variables into dummy variables
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model's performance
pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

# Save the trained model and feature names
joblib.dump(model, "../student-performance-predictor/models/model.pkl")
joblib.dump(X.columns, "../student-performance-predictor/models/features.pkl")