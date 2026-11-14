# F2) Streamlit web application for medical insurance charge prediction

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="🏥")

@st.cache_resource
def load_saved_model():
    return joblib.load("best_insurance_model.pkl")

saved = load_saved_model()
model = saved["model"]
model_name = saved["model_name"]
feature_columns = saved["columns"]
bmi_low = saved["bmi_low"]
bmi_high = saved["bmi_high"]

st.title("Medical Insurance Cost Predictor")
st.write("Enter the information below to estimate medical insurance charges.")
st.caption(f"Selected model: {model_name}")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=64, value=30)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=27.0, step=0.1)
    children = st.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)

with col2:
    sex = st.selectbox("Sex", ["female", "male"])
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

if st.button("Predict charges", type="primary"):
    # Apply the same BMI capping limits used during preprocessing.
    bmi_adjusted = min(max(bmi, bmi_low), bmi_high)

    # Match the encoding in Section B:
    # sex and smoker are 0/1; northeast is the dropped region category.
    new_customer = {
        "age": age,
        "bmi": bmi_adjusted,
        "children": children,
        "sex": 0 if sex == "female" else 1,
        "smoker": 0 if smoker == "no" else 1,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0
    }

    input_df = pd.DataFrame([new_customer]).reindex(
        columns=feature_columns, fill_value=0
    )
    predicted_charges = model.predict(input_df)[0]

    st.success(f"Estimated insurance charges: ${predicted_charges:,.2f}")

    if predicted_charges < 10000:
        st.info("Estimated category: Low")
    elif predicted_charges <= 20000:
        st.warning("Estimated category: Medium")
    else:
        st.error("Estimated category: High")

st.caption("This result is an estimate and should not be treated as an official insurance quote.")
