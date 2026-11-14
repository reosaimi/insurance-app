# F2) Streamlit web application for medical insurance charge prediction

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="🏥", layout="centered")

# ---------- Custom style (colors, cards, banner) ----------
st.markdown(
    """
    <style>
    .hero {background: linear-gradient(135deg, #0f766e 0%, #2563eb 100%);
           padding: 28px 26px; border-radius: 20px; margin-bottom: 22px;}
    .hero-title {font-size: 2.1rem; font-weight: 800; color: #ffffff; line-height: 1.2;}
    .hero-sub {color: #e0f2fe; margin-top: 8px; font-size: 1.02rem;}
    .result-card {border-radius: 20px; padding: 26px 18px; text-align: center; margin: 14px 0 10px 0;
                  border: 2px solid var(--c); background: rgba(128,128,128,0.10);}
    .result-label {opacity: 0.75; font-size: 0.95rem; letter-spacing: 0.04em;}
    .result-amount {font-size: 3rem; font-weight: 800; margin: 4px 0 10px 0;}
    .badge {display: inline-block; padding: 5px 18px; border-radius: 999px;
            color: #ffffff; font-weight: 700; background: var(--c);}
    .small-note {opacity: 0.65; font-size: 0.85rem; margin-top: 18px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Load the saved model ----------
@st.cache_resource
def load_saved_model():
    return joblib.load("best_insurance_model.pkl")

saved = load_saved_model()
model = saved["model"]
model_name = saved["model_name"]
feature_columns = saved["columns"]
bmi_low = saved["bmi_low"]
bmi_high = saved["bmi_high"]

DATASET_AVERAGE = 13279.0   # average charges in the cleaned dataset (from our EDA)
MAX_CHARGE = 65000.0        # highest charge in the dataset is about 63,770 (used for the gauge)

# ---------- Helper functions ----------
def bmi_label(value):
    if value < 18.5:
        return "Underweight"
    if value < 25:
        return "Normal"
    if value < 30:
        return "Overweight"
    return "Obese"

def get_category(value):
    if value < 10000:
        return "Low", "#16a34a"
    if value <= 20000:
        return "Medium", "#f59e0b"
    return "High", "#dc2626"

def predict(age, bmi, children, sex, smoker, region):
    # Same encoding and BMI capping used during preprocessing
    row = {
        "age": age,
        "bmi": min(max(bmi, bmi_low), bmi_high),
        "children": children,
        "sex": 0 if sex == "female" else 1,
        "smoker": 0 if smoker == "no" else 1,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }
    df_row = pd.DataFrame([row]).reindex(columns=feature_columns, fill_value=0)
    return float(model.predict(df_row)[0])

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About the model")
    st.write(f"**Model:** {model_name}")
    st.write("**R² (test set):** 0.899")
    st.write("**MAE:** about $2,450")
    st.write("**Data:** 1,337 records (Kaggle medical insurance dataset)")
    st.divider()
    st.caption("DS111 Project - Medical Insurance Analysis")

# ---------- Header ----------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🏥 Medical Insurance Cost Predictor</div>
        <div class="hero-sub">Enter the customer details and get an instant estimate of the yearly insurance charges.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_factors = st.tabs(["🔮 Prediction", "📊 What affects the cost?"])

# ---------- Tab 1: prediction ----------
with tab_predict:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", min_value=18, max_value=64, value=30)
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=27.0, step=0.1)
            children = st.number_input("Number of children", min_value=0, max_value=10, value=0, step=1)
        with col2:
            sex = st.selectbox("Sex", ["female", "male"])
            smoker = st.selectbox("Smoker", ["no", "yes"])
            region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
        submitted = st.form_submit_button("Predict charges", type="primary", use_container_width=True)

    if submitted:
        prediction = predict(age, bmi, children, sex, smoker, region)
        category, color = get_category(prediction)

        # Main result card
        st.markdown(
            f"""
            <div class="result-card" style="--c:{color}">
                <div class="result-label">ESTIMATED YEARLY CHARGES</div>
                <div class="result-amount">${prediction:,.2f}</div>
                <span class="badge">{category} cost</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Gauge: where this estimate sits between $0 and the maximum
        st.progress(min(prediction / MAX_CHARGE, 1.0))
        st.caption("Position of this estimate in the range of charges ($0 to $65,000)")

        # Key numbers
        m1, m2, m3 = st.columns(3)
        m1.metric("Monthly estimate", f"${prediction / 12:,.0f}")
        m2.metric("BMI status", bmi_label(bmi))
        m3.metric(
            "vs dataset average",
            f"${DATASET_AVERAGE:,.0f}",
            delta=f"{prediction - DATASET_AVERAGE:+,.0f}",
            delta_color="inverse",
        )

        # Smoking insight (smoking was the strongest factor in our EDA)
        if smoker == "yes":
            other = predict(age, bmi, children, sex, "no", region)
            st.info(f"🚭 If this person did not smoke, the estimate would be about ${other:,.0f} "
                    f"(${prediction - other:,.0f} less per year).")
        else:
            other = predict(age, bmi, children, sex, "yes", region)
            st.warning(f"🚬 If this person smoked, the estimate would rise to about ${other:,.0f} "
                       f"(${other - prediction:,.0f} more per year).")

        st.markdown(
            '<div class="small-note">This result is an estimate and should not be treated '
            'as an official insurance quote.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Fill in the information above and press **Predict charges**.")

# ---------- Tab 2: feature importance ----------
with tab_factors:
    st.subheader("Feature importance")
    st.write("How much each input contributes to the model's predictions "
             "(higher means more influence on the insurance charges).")
    if hasattr(model, "feature_importances_"):
        names = {
            "age": "Age", "sex": "Sex", "bmi": "BMI", "children": "Children",
            "smoker": "Smoker", "region_northwest": "Region: northwest",
            "region_southeast": "Region: southeast", "region_southwest": "Region: southwest",
        }
        importance = pd.Series(model.feature_importances_, index=feature_columns)
        importance.index = [names.get(c, c) for c in importance.index]
        st.bar_chart(importance.sort_values(ascending=False))
        st.caption("Smoking status is by far the most important factor, followed by BMI and age. "
                   "This matches the correlation results in our exploratory data analysis.")
    else:
        st.write("Feature importance is not available for this model.")
