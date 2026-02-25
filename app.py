
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and columns
model = joblib.load('loan_model_lite.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="Mortgage Risk AI", page_icon="🏦")

st.title("🏦 Mortgage Risk Underwriting Portal")
st.markdown("---")
st.write("Enter applicant details below to calculate the probability of default.")

# Creating input fields for the key features
col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Annual Income ($)", min_value=0, value=50000)
    loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, value=200000)
    credit_score = st.slider("Credit Score", 300, 850, 700)

with col2:
    property_value = st.number_input("Property Value ($)", min_value=0, value=250000)
    loan_term = st.selectbox("Loan Term (Months)", [120, 180, 240, 360])
    ltv = (loan_amount / property_value) * 100 if property_value > 0 else 0
    st.info(f"Calculated Loan-to-Value (LTV): {ltv:.2f}%")

if st.button("Analyze Risk", use_container_width=True):
    # Create a template dataframe with all 0s
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # Fill in the numerical values the user provided
    # (Note: These names must match your original dataset column names exactly)
    if 'income' in input_df.columns: input_df['income'] = income
    if 'loan_amount' in input_df.columns: input_df['loan_amount'] = loan_amount
    if 'LTV' in input_df.columns: input_df['LTV'] = ltv
    if 'Credit_Score' in input_df.columns: input_df['Credit_Score'] = credit_score

    # Make prediction
    prediction_prob = model.predict_proba(input_df)[0][1]
    
    st.markdown("### **Result:**")
    if prediction_prob >= 0.4:
        st.error(f"❌ **HIGH RISK** (Probability of Default: {prediction_prob:.2f})")
        st.write("Decision: Manual Review Required / Potential Rejection.")
    else:
        st.success(f"✅ **LOW RISK** (Probability of Default: {prediction_prob:.2f})")
        st.write("Decision: Recommended for Approval.")
