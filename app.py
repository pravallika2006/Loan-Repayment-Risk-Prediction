import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the saved model and the exact columns used during training
try:
    model = joblib.load('loan_model_lite.pkl')
    model_columns = joblib.load('model_columns.pkl')
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

st.set_page_config(page_title="Mortgage Risk AI", page_icon="🏦")

st.title("🏦 Mortgage Risk Underwriting Portal")
st.markdown("---")
st.write("Enter applicant details below to calculate the probability of default.")

# Creating input fields for the key features
col1, col2 = st.columns(2)

with col1:
    # Scale note: Use full dollar amounts (e.g., 50000)
    income = st.number_input("Annual Income ($)", min_value=0, value=65000)
    loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=250000)
    credit_score = st.slider("Credit Score", 300, 850, 720)

with col2:
    property_value = st.number_input("Property Value ($)", min_value=1, value=350000)
    # loan_term = st.selectbox("Loan Term (Months)", [120, 180, 240, 360]) # Optional based on your model
    
    # Critical Calculation: LTV is a top predictor in your project
    ltv = (loan_amount / property_value) * 100
    st.info(f"Calculated Loan-to-Value (LTV): {ltv:.2f}%")

if st.button("Analyze Risk", use_container_width=True):
    # 1. Create a template dataframe with a 'neutral' baseline (0.1 instead of 0)
    # This prevents the model from seeing the applicant as having 'zero' for everything
    input_df = pd.DataFrame(0.1, index=[0], columns=model_columns)
    
    # 2. Map user inputs to the correct column names from your training set
    # Note: These must match your CSV header names exactly (case-sensitive)
    mapping = {
        'income': income,
        'loan_amount': loan_amount,
        'LTV': ltv,
        'Credit_Score': credit_score
    }
    
    for col_name, value in mapping.items():
        if col_name in input_df.columns:
            input_df[col_name] = value

    # 3. Make prediction using the 0.4 threshold from your Case Study
    prediction_prob = model.predict_proba(input_df)[0][1]
    
    st.markdown("### **Result:**")
    if prediction_prob >= 0.4:
        st.error(f"❌ **HIGH RISK** (Probability of Default: {prediction_prob:.2f})")
        st.write("Decision: Manual Review Required / Potential Rejection.")
        st.warning("Factors contributing to risk: High LTV or relatively low Credit Score for the requested amount.")
    else:
        st.success(f"✅ **LOW RISK** (Probability of Default: {prediction_prob:.2f})")
        st.write("Decision: Recommended for Approval.")
