#🩺 Heart Disease Prediction App using Machine Learning

🚀 Overview

This project is a Machine Learning-based Web App built with Streamlit that predicts whether a person is at high or low risk of heart disease based on various medical attributes such as age, blood pressure, cholesterol, etc.

The backend model is a K-Nearest Neighbors (KNN) classifier trained on the Heart Disease Dataset (heart.csv).
It is fully integrated with data preprocessing, feature scaling, and user-friendly input fields for real-time prediction.


📊 Dataset
File: heart.csv

Total Samples: 918

Features: 16

Target Variable: HeartDisease (1 = High Risk, 0 = Low Risk)



💻 Streamlit Web App

The app interface allows users to input personal and medical information and get instant predictions.

Files:

app.py → Streamlit frontend

KNN_heart.pkl, scaler.pkl, columns.pkl → Model and preprocessing files



🧩 App Workflow

User enters input details (age, cholesterol, blood pressure, etc.)

App transforms inputs → scaled and encoded form

Model predicts:

🚨 High Risk of Heart Disease

✅ Low Risk of Heart Disease



📁 Project Structure

📦 Heart-Disease-Prediction

├── app.py

├── heart.csv

├── KNN_heart.pkl

├── scaler.pkl

├── columns.pkl

├── README.md

└── requirements.txt



📦 requirements.txt

streamlit

pandas

numpy

scikit-learn

joblib

matplotlib

seaborn



🧑‍💻 Author

Asim Raza

💡 Machine Learning Enthusiast | Python Developer

📧 Email: (asimrazayou@gmail.com)



