import joblib
import pandas as pd

# Load the trained model and feature names
model = joblib.load("../student-performance-predictor/models/model.pkl")
feature_names = joblib.load("../student-performance-predictor/models/features.pkl")

# Prepare input data for prediction
def prepare_input(data):
    # Ensure the input data has the same features as the model
    input_df = pd.DataFrame(data, columns=feature_names)
    input_df = input_df.fillna(input_df.mean(numeric_only=True))
    input_df = pd.get_dummies(input_df, drop_first=True)
    return input_df

# Make predictions
def predict(data):
    input_data = prepare_input(data)
    predictions = model.predict(input_data)
    return predictions

# Example usage
if __name__ == "__main__":
    # Replace with actual input data for prediction
    sample_data = {
        # Add feature values here
    }
    result = predict([sample_data])
    print("Predicted:", result)