from cairo import Path
import streamlit as st
import pandas as pd
import joblib

# Load model + features
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / ".."/"student-performance-predictor" / "models" / "model.pkl"
model = joblib.load(MODEL_PATH)

FEATURE_PATH = BASE_DIR / ".."/"student-performance-predictor" / "models" / "features.pkl"
features = joblib.load(FEATURE_PATH)

st.title("🎓 Student Performance Predictor")

st.write("Enter student details:")

importance = model.feature_importances_
feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

st.subheader("Top Features")
st.bar_chart(feat_df.set_index("Feature").head(10))


# ---- SIMPLE INPUTS (you can expand later) ----
with st.sidebar:
    st.header("Input Features")
    
    age = st.number_input("Age", 10, 100)
    attendance = st.slider("Attendance Rate (%)", 0, 100)
    study_hours = st.slider("Study Hours per Week", 0, 60)

    gender = st.selectbox("Gender", ["Male", "Female    "])
    internet = st.selectbox("Internet Access", ["Yes", "No"])

st.info("This model predicts student performance based on study habits and demographics.")

# ---- CREATE EMPTY DATAFRAME WITH CORRECT STRUCTURE ----
input_data = pd.DataFrame(columns=features)

# initialize all values to 0
input_data.loc[0] = 0


# fill numeric
if "age" in input_data.columns:
    input_data["age"] = age

if "attendance_rate" in input_data.columns:
    input_data["attendance_rate"] = attendance

if "study_hours_per_week" in input_data.columns:
    input_data["study_hours_per_week"] = study_hours

    
# fill categorical (one-hot)
if "gender_Male" in input_data.columns:
    input_data["gender_Male"] = 1 if gender == "Male" else 0

if "internet_access_Yes" in input_data.columns:
    input_data["internet_access_Yes"] = 1 if internet == "Yes" else 0


# ---- PREDICT ----
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Score: {prediction[0]:.2f}")


importance = model.feature_importances_
feat_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

st.subheader("Top Features")
st.bar_chart(feat_df.set_index("Feature").head(10))