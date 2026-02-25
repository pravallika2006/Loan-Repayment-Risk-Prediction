import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load files
model = joblib.load('loan_model_lite.pkl')
model_columns = joblib.load('model_columns.pkl')

# Baseline 'Safe' data you provided
SAFE_BASELINE = {'loan_amount': 334595, 'term': 335, 'property_value': 504388, 'income': 7044, 'Credit_Score': 699, 'age': 49, 'LTV': 71, 'loan_limit_ncf': 0.05, 'Gender_Joint': 0.29, 'Gender_Male': 0.27, 'Gender_Sex Not Available': 0.24, 'approv_in_adv_pre': 0.16, 'loan_type_type2': 0.12, 'loan_type_type3': 0.09, 'loan_purpose_p2': 0.01, 'loan_purpose_p3': 0.37, 'loan_purpose_p4': 0.37, 'Credit_Worthiness_l2': 0.03, 'open_credit_opc': 0.004, 'business_or_commercial_nob/c': 0.87, 'Neg_ammortization_not_neg': 0.92, 'interest_only_not_int': 0.95, 'lump_sum_payment_not_lpsm': 0.99, 'construction_type_sb': 1.0, 'occupancy_type_pr': 0.93, 'occupancy_type_sr': 0.01, 'Secured_by_land': 0.0, 'total_units_2U': 0.007, 'total_units_3U': 0.002, 'total_units_4U': 0.001, 'credit_type_CRIF': 0.33, 'credit_type_EQUI': 0.0, 'credit_type_EXP': 0.30, 'co-applicant_credit_type_EXP': 0.46, 'submission_of_application_to_inst': 0.61, 'Region_North-East': 0.007, 'Region_central': 0.05, 'Region_south': 0.42, 'Security_Type_direct': 1.0}

st.title("🏦 Mortgage Risk Underwriting Portal")

col1, col2 = st.columns(2)
with col1:
    income = st.number_input("Monthly Income", value=7000)
    loan_amount = st.number_input("Loan Amount", value=300000)
    credit_score = st.slider("Credit Score", 300, 850, 700)

with col2:
    property_value = st.number_input("Property Value", value=500000)
    ltv = (loan_amount / property_value) * 100
    st.info(f"LTV: {ltv:.2f}%")

if st.button("Analyze Risk"):
    # 1. Start with the 'Safe' baseline instead of zeros
    input_df = pd.DataFrame([SAFE_BASELINE])
    
    # 2. Update the baseline with user inputs
    input_df['income'] = income
    input_df['loan_amount'] = loan_amount
    input_df['property_value'] = property_value
    input_df['Credit_Score'] = credit_score
    input_df['LTV'] = ltv
    
    # 3. Ensure columns match the model exactly
    input_df = input_df[model_columns]
    
    # 4. Predict
    prob = model.predict_proba(input_df)[0][1]
    
    if prob < 0.4:
        st.success(f"✅ LOW RISK (Probability: {prob:.2f})")
    else:
        st.error(f"❌ HIGH RISK (Probability: {prob:.2f})")
