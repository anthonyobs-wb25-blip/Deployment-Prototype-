
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load assets
model = pickle.load(open('random_forest_classifier.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

st.title("Obesity Level Predictor")
st.write("Enter your details to determine your weight category.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.number_input("Age", min_value=1.0, max_value=100.0, value=25.0)
    height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.70)
    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=70.0)
    family_history = st.selectbox("Family history with overweight?", ["yes", "no"])
    favc = st.selectbox("High caloric food frequently?", ["yes", "no"])
    fcvc = st.slider("Frequency of vegetable consumption", 1.0, 3.0, 2.0)
    ncp = st.slider("Number of main meals", 1.0, 4.0, 3.0)

with col2:
    caec = st.selectbox("Food between meals", ["Sometimes", "Frequently", "Always", "no"])
    smoke = st.selectbox("Do you smoke?", ["yes", "no"])
    ch2o = st.slider("Daily water intake (Liters)", 1.0, 3.0, 2.0)
    scc = st.selectbox("Monitor calories?", ["yes", "no"])
    faf = st.slider("Physical activity frequency", 0.0, 3.0, 1.0)
    tue = st.slider("Time using technology devices", 0.0, 2.0, 1.0)
    calc = st.selectbox("Alcohol consumption", ["Sometimes", "Frequently", "Always", "no"])
    mtrans = st.selectbox("Transportation used", ["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

if st.button("Predict Weight Class"):
    # 1. Calculate BMI
    bmi = weight / (height ** 2)
    
    # 2. Match the exact feature columns used during training
    input_data = pd.DataFrame({
        'Age': [age], 'Height': [height], 'Weight': [weight], 
        'FCVC': [fcvc], 'NCP': [ncp], 'CH2O': [ch2o], 
        'FAF': [faf], 'TUE': [tue], 'BMI': [bmi],
        'Gender_Male': [1 if gender == 'Male' else 0],
        'family_history_with_overweight_yes': [1 if family_history == 'yes' else 0],
        'FAVC_yes': [1 if favc == 'yes' else 0],
        'CAEC_Frequently': [1 if caec == 'Frequently' else 0],
        'CAEC_Sometimes': [1 if caec == 'Sometimes' else 0],
        'CAEC_no': [1 if caec == 'no' else 0],
        'SMOKE_yes': [1 if smoke == 'yes' else 0],
        'SCC_yes': [1 if scc == 'yes' else 0],
        'CALC_Frequently': [1 if calc == 'Frequently' else 0],
        'CALC_Sometimes': [1 if calc == 'Sometimes' else 0],
        'CALC_no': [1 if calc == 'no' else 0],
        'MTRANS_Bike': [1 if mtrans == 'Bike' else 0],
        'MTRANS_Motorbike': [1 if mtrans == 'Motorbike' else 0],
        'MTRANS_Public_Transportation': [1 if mtrans == 'Public_Transportation' else 0],
        'MTRANS_Walking': [1 if mtrans == 'Walking' else 0]
    })

    # 3. Scale
    X_scaled = scaler.transform(input_data.values)

    # 4. Predict
    prediction = model.predict(X_scaled)
    result = le.inverse_transform(prediction)

    st.success(f"Predicted Weight Category: {result[0]}")
    st.info(f"Calculated BMI: {bmi:.2f}")
