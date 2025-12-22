import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from ckd_model import (
    predict_proba,
    MODEL_SCALER,
    MODEL_COLUMNS,
    MODEL_ACCURACY
)
scaler = MODEL_SCALER
columns = MODEL_COLUMNS
accuracy = MODEL_ACCURACY



# ===============================
# Theme Switch
# ===============================
mode = st.sidebar.radio("🎨 Theme Mode", ["🌞 Light Mode", "🌙 Dark Mode"])

if mode == "🌙 Dark Mode":
    bg = "#0f172a"
    card = "#1e293b"
    text = "#e5e7eb"
else:
    bg = "#f1f5f9"
    card = "#ffffff"
    text = "#0f172a"

# ===============================
# Page Config
# ===============================
st.set_page_config("CKD Smart Predictor", "🩺", "wide")

st.markdown(f"""
<style>
.stApp {{
    background-color: {bg};
    color: {text};
}}
.card {{
    background-color: {card};
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.15);
    margin-bottom: 20px;
}}
.stButton>button {{
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    height: 3em;
    border-radius: 14px;
    font-size: 18px;
    font-weight: bold;
}}
</style>
""", unsafe_allow_html=True)

# ===============================
# Header
# ===============================
st.markdown("## 🩺 Chronic Kidney Disease Predictor")
st.markdown("### Smart Medical Decision Support System")
st.markdown("---")

# ===============================
# Inputs
# ===============================
st.markdown("## 🧪 Patient Medical Data")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    age = st.number_input("Age", 1, 120, 50)
    bp = st.number_input("Blood Pressure", 50, 200, 80)
    sg = st.selectbox("Specific Gravity",[1.005,1.010,1.015,1.020,1.025])
    al = st.selectbox("Albumin",[0,1,2,3,4,5])
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    su = st.selectbox("Sugar",[0,1,2,3,4,5])
    rbc = st.selectbox("RBC",[0,1])
    pcv = st.number_input("PCV",10,60,40)
    wc = st.number_input("WBC",2000,20000,8000)
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    hemo = st.number_input("Hemoglobin",3.0,20.0,14.0,0.1)
    htn = st.selectbox("Hypertension",[0,1])
    dm = st.selectbox("Diabetes",[0,1])
    appet = st.selectbox("Appetite",[0,1])
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# Prepare Input
# ===============================
input_df = pd.DataFrame({
    "age":[age],"bp":[bp],"sg":[sg],"al":[al],"su":[su],
    "rbc":[rbc],"pcv":[pcv],"wc":[wc],"hemo":[hemo],
    "htn":[htn],"dm":[dm],"appet":[appet]
})

for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[columns]
input_scaled = scaler.transform(input_df)

# ===============================
# Prediction
# ===============================
st.markdown("---")
st.markdown("## 📊 Prediction Result")

if st.button("🔍 Analyze Patient", use_container_width=True):
    prob = predict_proba(input_scaled)[0]
    risk = prob * 100

    if risk >= 60:
        st.error(f"⚠️ High Risk of CKD ({risk:.2f}%)")
    else:
        st.success(f"✅ Low Risk ({risk:.2f}%)")

    # ===============================
    # Gauge Chart
    # ===============================
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        title={"text": "CKD Risk Level (%)"},
        gauge={
            "axis": {"range": [0,100]},
            "bar": {"color": "#2563eb"},
            "steps": [
                {"range":[0,40],"color":"#22c55e"},
                {"range":[40,60],"color":"#eab308"},
                {"range":[60,100],"color":"#ef4444"}
            ],
        }
    ))
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# Model Info
# ===============================
st.markdown("---")
st.markdown("## 📈 Model Information")

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Accuracy", f"{accuracy*100:.2f}%")
with m2:
    st.metric("Model Type", "Logistic Regression")
with m3:
    st.metric("Implementation", "From Scratch (NumPy)")

st.markdown("""
⚠️ **Disclaimer:**  
Educational use only – not a medical diagnosis.
""")
