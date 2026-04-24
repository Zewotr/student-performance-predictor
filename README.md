# 🎓 Student Performance Predictor

## 📌 Overview

This project is an end-to-end Machine Learning application that predicts student academic performance based on various factors such as study time, attendance, and previous scores.

The goal is to identify patterns that influence student success and provide insights for early intervention.

---

## 🧠 Problem Statement

Educational institutions often struggle to identify students at risk of underperforming.
This project uses Machine Learning to predict student performance, helping educators take proactive measures.

---

## ⚙️ Tech Stack

- **Language:** Python
- **Libraries:** scikit-learn, pandas, numpy, matplotlib
- **Model:** Linear Regression / Classification (depending on your version)
- **Web Framework:** Streamlit
- **Tools:** Jupyter Notebook, Git, GitHub

---

## 📊 Dataset

- Student performance dataset (CSV)
- Features may include:
  - Study Hours
  - Attendance
  - Previous Scores
  - Sleep Hours
  - Participation Level

---

## 🏗️ Project Structure

```
student-performance-predictor/
│
├── data/
│   └── student_performance.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   ├── model.pkl
│   └── features.pkl
│
├── .gitignore
├── requirements.txt
├── README.md
```

---

## 🧪 Model Training

Steps followed:

1. Data cleaning and preprocessing
2. Feature selection
3. Train-test split
4. Model training
5. Model evaluation

---

## 📈 Model Performance

- R² Score: 0.76

---

## 💻 Running the Project Locally

### 1. Clone repository

```
git clone https://github.com/Zewotr/student-performance-predictor.git
cd student-performance-predictor
```

### 2. Create virtual environment

```
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Run Streamlit app

```
jupyter notebook
streamlit run streamlit_app.py
```

---

## 🎯 Features

- Predict student performance based on input features
- Interactive web interface
- Fast and lightweight ML model
- Useful for educational insights

---

## 🔥 Future Improvements

- Use advanced models (Random Forest, XGBoost)
- Add feature importance visualization
- Improve dataset size and quality
- Deploy online for public use
- Add dashboard with analytics

---

## 👨‍💻 Author

**Zewotr Lamesginew**

- GitHub: https://github.com/Zewotr
- LinkedIn: www.linkedin.com/in/zewotr-lamesginew-7b991137a

---

## Dataset source

Dataset is found from https://www.kaggle.com/datasets/mubashirsidiki/student-academic-performance-500-students.

And it is really helpful,thank you.

## 📌 Notes

This project demonstrates practical application of Machine Learning in education and serves as part of my ML portfolio.
