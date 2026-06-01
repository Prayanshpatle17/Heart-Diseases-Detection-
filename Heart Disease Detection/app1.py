import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("heart_scaler.pkl")

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction System")

st.write("Enter patient details below:")

# Numerical Inputs
age = st.number_input("Age", 20, 100, 55)

resting_blood_pressure = st.number_input(
    "Resting Blood Pressure",
    80,
    250,
    130
)

cholesterol = st.number_input(
    "Cholesterol",
    100,
    700,
    250
)

max_heart_rate_achieved = st.number_input(
    "Maximum Heart Rate Achieved",
    50,
    250,
    150
)

st_depression = st.number_input(
    "ST Depression",
    0.0,
    10.0,
    1.0
)

num_major_vessels = st.selectbox(
    "Number of Major Vessels",
    [0, 1, 2, 3]
)

# Categorical Inputs

sex = st.selectbox(
    "Sex",
    ["female", "male"]
)

chest_pain_type = st.selectbox(
    "Chest Pain Type",
    [
        "asymptomatic",
        "atypical angina",
        "non-anginal pain",
        "typical angina"
    ]
)

fasting_blood_sugar = st.selectbox(
    "Fasting Blood Sugar",
    [
        "greater than 120mg/ml",
        "lower than 120mg/ml"
    ]
)

resting_electrocardiogram = st.selectbox(
    "Resting ECG",
    [
        "ST-T wave abnormality",
        "left ventricular hypertrophy",
        "normal"
    ]
)

exercise_induced_angina = st.selectbox(
    "Exercise Induced Angina",
    [
        "no",
        "yes"
    ]
)

st_slope = st.selectbox(
    "ST Slope",
    [
        "downsloping",
        "flat",
        "upsloping"
    ]
)

thalassemia = st.selectbox(
    "Thalassemia",
    [
        "fixed defect",
        "normal",
        "reversable defect"
    ]
)

if st.button("Predict"):

    patient = {
        'age': age,
        'resting_blood_pressure': resting_blood_pressure,
        'cholesterol': cholesterol,
        'max_heart_rate_achieved': max_heart_rate_achieved,
        'st_depression': st_depression,
        'num_major_vessels': num_major_vessels,

        'sex_male': 1 if sex == "male" else 0,

        'chest_pain_type_atypical angina':
            1 if chest_pain_type == "atypical angina" else 0,

        'chest_pain_type_non-anginal pain':
            1 if chest_pain_type == "non-anginal pain" else 0,

        'chest_pain_type_typical angina':
            1 if chest_pain_type == "typical angina" else 0,

        'fasting_blood_sugar_lower than 120mg/ml':
            1 if fasting_blood_sugar == "lower than 120mg/ml" else 0,

        'resting_electrocardiogram_left ventricular hypertrophy':
            1 if resting_electrocardiogram ==
            "left ventricular hypertrophy" else 0,

        'resting_electrocardiogram_normal':
            1 if resting_electrocardiogram == "normal" else 0,

        'exercise_induced_angina_yes':
            1 if exercise_induced_angina == "yes" else 0,

        'st_slope_flat':
            1 if st_slope == "flat" else 0,

        'st_slope_upsloping':
            1 if st_slope == "upsloping" else 0,

        'thalassemia_normal':
            1 if thalassemia == "normal" else 0,

        'thalassemia_reversable defect':
            1 if thalassemia == "reversable defect" else 0
    }

    patient_df = pd.DataFrame([patient])

    patient_scaled = scaler.transform(patient_df)

    prediction = model.predict(patient_scaled)[0]

    probability = (
        model.predict_proba(patient_scaled)[0][1]
        * 100
    )

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(
            f"⚠ High Risk of Heart Disease\n\n"
            f"Risk Probability: {probability:.2f}%"
        )
    else:
        st.success(
            f"✅ Low Risk of Heart Disease\n\n"
            f"Risk Probability: {probability:.2f}%"
        )
        