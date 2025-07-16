import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('loan_eligibility_final_model.pkl', 'rb'))

st.title("Loan Approval Prediction")
st.markdown("Welcome to the Streamlit app for predicting loan approval! 🏦")

# User input
no_of_dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])
income_annum = st.number_input("Annual Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (in days)", min_value=0)
cibil_score = st.number_input("CIBIL Score", min_value=0)
residential_assets_value = st.number_input("Residential Assets Value", min_value=0)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0)

# Convert categorical to numeric
dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
education_map = {"Graduate": 1, "Not Graduate": 0}
self_employed_map = {"No": 0, "Yes": 1}

input_list = [
    dependents_map[no_of_dependents],
    education_map[education],
    self_employed_map[self_employed],
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
]

input_array = np.array([input_list])
st.write("🔍 Model Input:", input_array)  # Debug input

if st.button("Predict"):
    prediction = model.predict(input_array)
    st.write("🔍 Model Prediction Output:", prediction[0])  # Debug output
    if prediction[0] == 1:
        st.success("✅ Loan is likely to be Approved!")
    else:
        st.error("❌ Loan is likely to be Rejected.")
