import streamlit as st
import pandas as pd
import joblib

# 1. Load Files
model = joblib.load('loan_model_lite.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("🏦 Final Mortgage Risk Test")

# 2. Inputs
income = st.number_input("Income", value=80000)
loan = st.number_input("Loan Amount", value=150000)
prop_value = st.number_input("Property Value", value=400000)
credit = st.slider("Credit Score", 300, 850, 750)

if st.button("Run Final Analysis"):
    # Calculate LTV
    ltv_val = (loan / prop_value) * 100
    
    # CREATE DATAFRAME - Matching exact case from your training
    # We will try both lowercase and Title case to be safe
    input_data = pd.DataFrame(0.1, index=[0], columns=model_columns)
    
    # Apply values to common variations of names
    for col in input_data.columns:
        c_low = col.lower()
        if c_low == 'income': input_data[col] = income
        if c_low == 'loan_amount': input_data[col] = loan
        if c_low == 'ltv': input_data[col] = ltv_val
        if c_low == 'credit_score': input_data[col] = credit

    # Predict
    prob = model.predict_proba(input_data)[0][1]
    
    if prob < 0.4:
        st.success(f"✅ LOW RISK (Score: {prob:.2f})")
    else:
        st.error(f"❌ HIGH RISK (Score: {prob:.2f})")

    # DEBUG SECTION - THIS WILL SHOW US THE PROBLEM
    with st.expander("Technical Debugging (Show this to me if it still fails)"):
        st.write("Columns model expects:", model_columns[:10]) # Show first 10
        st.write("Values being sent:", input_data[['income', 'loan_amount', 'LTV', 'Credit_Score'] if 'income' in input_data.columns else input_data.columns[:4]])
