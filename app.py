import streamlit as st
import pandas as pd
import pickle
import base64
import matplotlib.pyplot as plt
import seaborn as sns

# === Load trained model ===
with open("LinearRegressor.pkl", "rb") as file:
    model = pickle.load(file)

# === Background image setup ===
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .glass {{
            background: rgba(0, 0, 0, 0.4);
            padding: 2rem;
            border-radius: 20px;
            width: 85%;
            max-width: 700px;
            margin: auto;
            color: white;
            box-shadow: 0 0 25px rgba(255, 255, 255, 0.25);
        }}
        .title {{
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            color: white;
            text-shadow: 2px 2px 4px #000;
        }}
        .prediction {{
            font-size: 1.8rem;
            font-weight: 600;
            text-align: center;
            margin-top: 1rem;
            color: #ffffff;
            text-shadow: 1px 1px 6px #000;
        }}
        .stButton > button {{
            padding: 0.5rem 1.2rem;
            border-radius: 10px;
            font-weight: bold;
            font-size: 1rem;
            border: none;
            color: white;
            background-color: #0074cc;
            transition: 0.2s ease-in-out;
        }}
        .stButton > button:hover {{
            background-color: #005fa3;
            transform: scale(1.03);
        }}
        </style>
    """, unsafe_allow_html=True)

# === Average region costs for chart ===
region_costs = {
    "northeast": 31500,
    "northwest": 27500,
    "southeast": 42000,
    "southwest": 26000
}

# === Streamlit Config ===
st.set_page_config(page_title="Insurance Cost Predictor", layout="centered")
set_background("background.jpg")

# === Sidebar Tips ===
st.sidebar.title("💡 Tips & Region Stats")
st.sidebar.markdown("""
- 🚭 Non-smokers pay significantly less.
- 👶 Children slightly affect premium.
- 🌍 Southeast region tends to be costliest.
- 💪 High BMI increases cost.
---
**Region-Wise Average Costs**
""")
st.sidebar.bar_chart(region_costs)

# === Page Title ===
st.markdown("<h1 class='title'>💰 Insurance Cost Predictor</h1>", unsafe_allow_html=True)

# === Form Box ===
st.markdown("<div class='glass'>", unsafe_allow_html=True)

with st.form("insurance_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)

    with col2:
        children = st.number_input("Children", min_value=0, max_value=10, value=1)
        smoker = st.selectbox("Smoker", ["yes", "no"])
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        submitted = st.form_submit_button("🔍 Predict")
    with col_btn2:
        reset = st.form_submit_button("🔄 Reset")

st.markdown("</div>", unsafe_allow_html=True)

# === Predict ===
if submitted:
    input_df = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    try:
        cost = model.predict(input_df)[0]
        formatted_cost = f"₹ {cost:,.2f}"
        st.markdown(f"<div class='prediction'>🎯 Estimated Insurance Cost: {formatted_cost}</div>", unsafe_allow_html=True)

        # Bar chart comparison
        st.subheader("📊 Your Prediction vs Region Average")
        comparison_df = pd.DataFrame({
            "Region": list(region_costs.keys()) + [region],
            "Cost": list(region_costs.values()) + [cost]
        })
        fig, ax = plt.subplots()
        sns.barplot(data=comparison_df, x="Region", y="Cost", palette="coolwarm", ax=ax)
        ax.set_title("Insurance Cost Comparison")
        st.pyplot(fig)

        # === CSV download ===
        result_df = input_df.copy()
        result_df["Predicted Cost"] = cost
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Prediction as CSV",
            data=csv,
            file_name="insurance_prediction.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Prediction Failed: {e}")

# === Reset ===
if reset:
    st.experimental_rerun()
