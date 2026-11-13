# Step 2: Streamlit web app to predict medical insurance charges

import joblib
import pandas as pd
import streamlit as st

# Page settings: browser tab title and icon
st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="🏥")

# Function that loads the saved model
# cache_resource makes the app load the model only once, which makes it faster
@st.cache_resource
def load_model():
    return joblib.load("rf_model.pkl")

# Load the model and the column order
saved = load_model()
model = saved["model"]
columns = saved["columns"]

# App title and description
st.title("🏥 Medical Insurance Cost Predictor")
st.write("This app uses our best-performing model (Random Forest, R² = 0.87) "
         "to estimate yearly medical insurance charges.")

st.subheader("Enter the customer information")

# Split the page into two columns to keep the layout organized
col1, col2 = st.columns(2)

# First column: numeric inputs
with col1:
    # Slider to choose the age (from 18 to 64, default value 30)
    age = st.slider("Age", 18, 64, 30)
    # Input box for the BMI (body mass index)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=27.0, step=0.1)
    # Input box for the number of children
    children = st.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)

# Second column: categorical inputs (dropdown lists)
with col2:
    sex = st.selectbox("Sex", ["female", "male"])
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Predict button: the code below runs when the user clicks it
if st.button("Predict charges", type="primary"):

    # Convert the user inputs into numbers using the same encoding as in training
    # Example: if sex is male then sex_male = 1, if female then 0
    # The northeast region was dropped (drop_first), so all region columns are 0 for it
    row = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }

    # Convert the inputs into a DataFrame and arrange the columns like in training
    X_new = pd.DataFrame([row])[columns]

    # The model predicts the insurance charges
    prediction = model.predict(X_new)[0]

    # Show the result to the user
    st.success(f"Estimated insurance charges: ${prediction:,.2f}")

    # Classify the cost (same classes used when we evaluated the models)
    if prediction < 10000:
        st.info("Cost category: Low (below 10,000)")
    elif prediction <= 20000:
        st.warning("Cost category: Medium (10,000 - 20,000)")
    else:
        st.error("Cost category: High (above 20,000)")

# Small note at the bottom of the page
st.caption("DS111 Project - Medical Insurance Analysis. Predictions are estimates only.")
