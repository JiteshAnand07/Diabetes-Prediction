# import streamlit as st
# import numpy as np
# import pandas as pd
# import joblib
# import os

# # ── Page config ─────────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="Diabetes Prediction | ML App",
#     page_icon="🩺",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ── Custom CSS ──────────────────────────────────────────────────────────────────
# st.markdown("""
# <style>
#     .main-header {
#         background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
#         color: white;
#         padding: 2rem;
#         border-radius: 12px;
#         text-align: center;
#         margin-bottom: 2rem;
#         box-shadow: 0 4px 15px rgba(0,0,0,0.3);
#     }
#     .risk-low    { background-color: #d4edda; color: #155724; border-left: 6px solid #28a745; }
#     .risk-medium { background-color: #fff3cd; color: #856404; border-left: 6px solid #ffc107; }
#     .risk-high   { background-color: #f8d7da; color: #721c24; border-left: 6px solid #dc3545; }
#     .risk-box    { padding: 1.2rem 1.5rem; border-radius: 8px; font-size: 1.1rem; font-weight: 600; margin-top: 1rem; }
#     .metric-card { background: #f8f9fa; border-radius: 8px; padding: 1rem; text-align: center; }
#     .footer      { text-align: center; color: #666; font-size: 0.85rem; margin-top: 3rem; padding: 1rem; border-top: 1px solid #eee; }
# </style>
# """, unsafe_allow_html=True)

# # ── Load model artifacts ─────────────────────────────────────────────────────────
# @st.cache_resource
# def load_artifacts():
#     model    = joblib.load('model_artifacts/model.pkl')
#     scaler   = joblib.load('model_artifacts/scaler.pkl')
#     features = joblib.load('model_artifacts/top_features.pkl')
#     return model, scaler, features

# try:
#     model, scaler, features = load_artifacts()
#     model_loaded = True
# except Exception as e:
#     model_loaded = False
#     model_error = str(e)

# # ── Header ───────────────────────────────────────────────────────────────────────
# st.markdown("""
# <div class="main-header">
#     <h1>🩺 Diabetes Risk Prediction System</h1>
#     <p style="font-size:1.1rem; opacity:0.85;">
#         AI-powered early diabetes screening using clinical biomarkers
#     </p>
#     <p style="font-size:0.9rem; opacity:0.65;">
#         Built with XGBoost · ROC-AUC ~0.97 · 100,000 patient records
#     </p>
# </div>
# """, unsafe_allow_html=True)

# # ── Sidebar ──────────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
#     st.markdown("## 📋 Patient Input Form")
#     st.markdown("---")

#     gender = st.selectbox("👤 Gender", ["Female", "Male"])
#     age    = st.slider("🎂 Age", min_value=1, max_value=80, value=45, step=1)

#     st.markdown("---")
#     st.markdown("**🏥 Medical History**")
#     hypertension  = st.checkbox("Hypertension")
#     heart_disease = st.checkbox("Heart Disease")

#     st.markdown("---")
#     st.markdown("**🚬 Lifestyle**")
#     smoking = st.selectbox("Smoking History",
#                             ["never", "No Info", "current", "former", "ever", "not current"])
#     bmi = st.slider("⚖️ BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)

#     st.markdown("---")
#     st.markdown("**🔬 Lab Results**")
#     hba1c   = st.slider("HbA1c Level (%)", min_value=3.5, max_value=9.0, value=5.5, step=0.1)
#     glucose = st.slider("Blood Glucose Level (mg/dL)", min_value=80, max_value=300, value=120, step=1)

#     predict_btn = st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True)

# # ── Main content ─────────────────────────────────────────────────────────────────
# col1, col2 = st.columns([1.2, 1])

# with col1:
#     st.markdown("### 📊 Patient Summary")
#     summary_data = {
#         "Parameter":  ["Gender", "Age", "BMI", "HbA1c Level", "Blood Glucose",
#                         "Hypertension", "Heart Disease", "Smoking History"],
#         "Value":      [gender, f"{age} yrs", f"{bmi:.1f}", f"{hba1c:.1f}%",
#                         f"{glucose} mg/dL", "Yes" if hypertension else "No",
#                         "Yes" if heart_disease else "No", smoking],
#         "Reference":  ["—", "18–65 normal", "18.5–24.9 normal", "< 5.7% normal",
#                         "70–99 mg/dL fasting", "—", "—", "—"]
#     }
#     st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

#     # Clinical reference
#     st.markdown("### 📚 Clinical Reference")
#     st.info("""
#     | HbA1c | Interpretation |
#     |-------|---------------|
#     | < 5.7% | Normal |
#     | 5.7–6.4% | Pre-diabetes |
#     | ≥ 6.5% | Diabetes |

#     | Blood Glucose (fasting) | Interpretation |
#     |------------------------|---------------|
#     | < 100 mg/dL | Normal |
#     | 100–125 mg/dL | Pre-diabetes |
#     | ≥ 126 mg/dL | Diabetes |
#     """)

# with col2:
#     st.markdown("### 🎯 Prediction Results")

#     if predict_btn:
#         if not model_loaded:
#             st.error(f"❌ Model not found. Please run the notebook first to generate model artifacts.\n\n`{model_error}`")
#         else:
#             # Build input
#             gender_enc = 1 if gender == "Male" else 0
#             smoke_cats = ['never', 'No Info', 'current', 'former', 'ever', 'not current']
#             smoke_cols = {f'smoke_{s}': 0 for s in smoke_cats}
#             smoke_cols[f'smoke_{smoking}'] = 1

#             input_dict = {
#                 'age': age,
#                 'hypertension': int(hypertension),
#                 'heart_disease': int(heart_disease),
#                 'bmi': bmi,
#                 'HbA1c_level': hba1c,
#                 'blood_glucose_level': glucose,
#                 'gender_enc': gender_enc,
#                 **smoke_cols
#             }
#             input_df = pd.DataFrame([input_dict])

#             # Scale only numeric columns
#             num_cols = ['age', 'hypertension', 'heart_disease', 'bmi',
#                         'HbA1c_level', 'blood_glucose_level', 'gender_enc']
#             try:
#                 input_scaled = input_df[features]
#             except KeyError:
#                 # Align columns
#                 for f in features:
#                     if f not in input_df.columns:
#                         input_df[f] = 0
#                 input_scaled = input_df[features]

#             proba    = model.predict_proba(input_scaled)[0][1]
#             pred     = int(proba >= 0.5)

#             # Risk category
#             if proba < 0.3:
#                 risk_label = "🟢 LOW RISK"
#                 risk_class = "risk-low"
#                 advice = "Your results suggest a low risk of diabetes. Maintain a healthy lifestyle with regular exercise and balanced diet."
#             elif proba < 0.6:
#                 risk_label = "🟡 MEDIUM RISK"
#                 risk_class = "risk-medium"
#                 advice = "You have a moderate risk of developing diabetes. Consult your doctor and consider lifestyle modifications."
#             else:
#                 risk_label = "🔴 HIGH RISK"
#                 risk_class = "risk-high"
#                 advice = "Your profile indicates high diabetes risk. Please consult a healthcare professional immediately for a comprehensive evaluation."

#             # Display
#             st.metric("Diabetes Probability", f"{proba*100:.1f}%",
#                        delta=f"{'⚠️ Diabetic' if pred==1 else '✅ Non-Diabetic'}")

#             st.progress(float(proba))

#             st.markdown(f"""
#             <div class="risk-box {risk_class}">
#                 <strong>{risk_label}</strong><br>
#                 {advice}
#             </div>
#             """, unsafe_allow_html=True)

#             st.markdown("---")
#             st.markdown("**🔬 Key Risk Factors Identified:**")
#             flags = []
#             if hba1c >= 6.5:   flags.append("🔴 HbA1c ≥ 6.5% (Diabetes range)")
#             elif hba1c >= 5.7: flags.append("🟡 HbA1c 5.7–6.4% (Pre-diabetes range)")
#             if glucose >= 126:  flags.append("🔴 Blood glucose ≥ 126 mg/dL (Diabetes range)")
#             elif glucose >= 100: flags.append("🟡 Blood glucose 100–125 mg/dL (Pre-diabetes range)")
#             if bmi >= 30:       flags.append("🟡 BMI ≥ 30 (Obese)")
#             if hypertension:    flags.append("⚠️ Hypertension present")
#             if heart_disease:   flags.append("⚠️ Heart disease present")
#             if age > 45:        flags.append("ℹ️ Age > 45 (increased risk group)")

#             if flags:
#                 for flag in flags:
#                     st.markdown(f"- {flag}")
#             else:
#                 st.success("No major risk flags detected.")

#     else:
#         st.info("👈 Fill in the patient details in the sidebar and click **Predict Diabetes Risk**")
#         st.markdown("""
#         **About This Application:**
#         - Trained on 100,000 patient records
#         - Uses XGBoost with hyperparameter tuning
#         - Achieves ~97% ROC-AUC on validation data
#         - Provides probability score and risk category
#         - ⚠️ For educational purposes only — not a substitute for medical advice
#         """)

# # ── Footer ───────────────────────────────────────────────────────────────────────
# st.markdown("""
# <div class="footer">
#     🩺 Diabetes Prediction System &nbsp;|&nbsp; Built with Streamlit + XGBoost &nbsp;|&nbsp;
#     ⚠️ <em>For educational/research purposes only. Not a substitute for professional medical advice.</em>
# </div>
# """, unsafe_allow_html=True)


import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Diabetes Prediction | ML App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        color: white; padding: 2rem; border-radius: 12px;
        text-align: center; margin-bottom: 2rem;
    }
    .risk-low    { background-color: #d4edda; color: #155724; border-left: 6px solid #28a745; }
    .risk-medium { background-color: #fff3cd; color: #856404; border-left: 6px solid #ffc107; }
    .risk-high   { background-color: #f8d7da; color: #721c24; border-left: 6px solid #dc3545; }
    .risk-box    { padding: 1.2rem 1.5rem; border-radius: 8px; font-size: 1.1rem; font-weight: 600; margin-top: 1rem; }
    .footer      { text-align: center; color: #666; font-size: 0.85rem; margin-top: 3rem; padding: 1rem; border-top: 1px solid #eee; }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts (no caching to avoid stale state) ──────────────────
model   = joblib.load('model_artifacts/model.pkl')
scaler  = joblib.load('model_artifacts/scaler.pkl')

SCALER_COLS = [
    'age', 'hypertension', 'heart_disease', 'bmi',
    'HbA1c_level', 'blood_glucose_level', 'gender_enc',
    'smoke_No Info', 'smoke_current', 'smoke_ever',
    'smoke_former', 'smoke_never', 'smoke_not current'
]

TOP_FEATURES = [
    'HbA1c_level', 'blood_glucose_level', 'age', 'bmi',
    'hypertension', 'smoke_No Info', 'heart_disease', 'smoke_former'
]

st.markdown("""
<div class="main-header">
    <h1>🩺 Diabetes Risk Prediction System</h1>
    <p style="font-size:1.1rem; opacity:0.85;">AI-powered early diabetes screening using clinical biomarkers</p>
    <p style="font-size:0.9rem; opacity:0.65;">Built with XGBoost · ROC-AUC ~0.97 · 100,000 patient records</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
    st.markdown("## 📋 Patient Input Form")
    st.markdown("---")
    gender        = st.selectbox("👤 Gender", ["Female", "Male"])
    age           = st.slider("🎂 Age", min_value=1, max_value=80, value=45, step=1)
    st.markdown("---")
    st.markdown("**🏥 Medical History**")
    hypertension  = st.checkbox("Hypertension")
    heart_disease = st.checkbox("Heart Disease")
    st.markdown("---")
    st.markdown("**🚬 Lifestyle**")
    smoking       = st.selectbox("Smoking History",
                                 ["never", "No Info", "current", "former", "ever", "not current"])
    bmi           = st.slider("⚖️ BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    st.markdown("---")
    st.markdown("**🔬 Lab Results**")
    hba1c         = st.slider("HbA1c Level (%)", min_value=3.5, max_value=9.0, value=5.5, step=0.1)
    glucose       = st.slider("Blood Glucose (mg/dL)", min_value=80, max_value=300, value=120, step=1)
    predict_btn   = st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown("### 📊 Patient Summary")
    st.dataframe(pd.DataFrame({
        "Parameter": ["Gender", "Age", "BMI", "HbA1c Level", "Blood Glucose",
                      "Hypertension", "Heart Disease", "Smoking History"],
        "Value":     [gender, f"{age} yrs", f"{bmi:.1f}", f"{hba1c:.1f}%",
                      f"{glucose} mg/dL", "Yes" if hypertension else "No",
                      "Yes" if heart_disease else "No", smoking],
        "Reference": ["—", "18–65 normal", "18.5–24.9 normal", "< 5.7% normal",
                      "70–99 mg/dL fasting", "—", "—", "—"]
    }), use_container_width=True, hide_index=True)

    st.markdown("### 📚 Clinical Reference")
    st.info("""
    | HbA1c | Interpretation |
    |-------|---------------|
    | < 5.7% | Normal |
    | 5.7–6.4% | Pre-diabetes |
    | ≥ 6.5% | Diabetes |

    | Blood Glucose (fasting) | Interpretation |
    |------------------------|---------------|
    | < 100 mg/dL | Normal |
    | 100–125 mg/dL | Pre-diabetes |
    | ≥ 126 mg/dL | Diabetes |
    """)

with col2:
    st.markdown("### 🎯 Prediction Results")

    if predict_btn:

        # ── Build input dict exactly like the debug cell ──────────────
        data = {
            'age':                 float(age),
            'hypertension':        float(int(hypertension)),
            'heart_disease':       float(int(heart_disease)),
            'bmi':                 float(bmi),
            'HbA1c_level':         float(hba1c),
            'blood_glucose_level': float(glucose),
            'gender_enc':          1.0 if gender == "Male" else 0.0,
            'smoke_No Info':       1.0 if smoking == 'No Info'      else 0.0,
            'smoke_current':       1.0 if smoking == 'current'      else 0.0,
            'smoke_ever':          1.0 if smoking == 'ever'         else 0.0,
            'smoke_former':        1.0 if smoking == 'former'       else 0.0,
            'smoke_never':         1.0 if smoking == 'never'        else 0.0,
            'smoke_not current':   1.0 if smoking == 'not current'  else 0.0,
        }

        # ── Scale ─────────────────────────────────────────────────────
        df     = pd.DataFrame([data])[SCALER_COLS]
        scaled = pd.DataFrame(scaler.transform(df), columns=SCALER_COLS)
        final  = scaled[TOP_FEATURES]

        # ── Predict ───────────────────────────────────────────────────
        proba  = model.predict_proba(final)[0][1]
        pred   = int(proba >= 0.5)

        # ── Risk category ─────────────────────────────────────────────
        if proba < 0.3:
            risk_label = "🟢 LOW RISK"
            risk_class = "risk-low"
            advice     = "Your results suggest a low risk of diabetes. Maintain a healthy lifestyle."
        elif proba < 0.6:
            risk_label = "🟡 MEDIUM RISK"
            risk_class = "risk-medium"
            advice     = "Moderate risk detected. Consult your doctor and consider lifestyle modifications."
        else:
            risk_label = "🔴 HIGH RISK"
            risk_class = "risk-high"
            advice     = "High diabetes risk detected. Please consult a healthcare professional immediately."

        st.metric("Diabetes Probability", f"{proba*100:.1f}%",
                  delta=f"{'⚠️ Diabetic' if pred == 1 else '✅ Non-Diabetic'}")
        st.progress(float(proba))
        st.markdown(f"""
        <div class="risk-box {risk_class}">
            <strong>{risk_label}</strong><br>{advice}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**🔬 Key Risk Factors:**")
        flags = []
        if hba1c >= 6.5:     flags.append("🔴 HbA1c ≥ 6.5% (Diabetes range)")
        elif hba1c >= 5.7:   flags.append("🟡 HbA1c 5.7–6.4% (Pre-diabetes)")
        if glucose >= 126:   flags.append("🔴 Blood glucose ≥ 126 mg/dL")
        elif glucose >= 100: flags.append("🟡 Blood glucose 100–125 mg/dL (Pre-diabetes)")
        if bmi >= 30:        flags.append("🟡 BMI ≥ 30 (Obese)")
        if hypertension:     flags.append("⚠️ Hypertension present")
        if heart_disease:    flags.append("⚠️ Heart disease present")
        if age > 45:         flags.append("ℹ️ Age > 45 (increased risk)")
        if flags:
            for flag in flags:
                st.markdown(f"- {flag}")
        else:
            st.success("No major risk flags detected.")

        # ── Debug info ────────────────────────────────────────────────
        with st.expander("🔍 Debug Info"):
            st.write("Raw input:", data)
            st.write("Scaled final input:", final)
            st.write(f"Raw probability: {proba:.6f}")

    else:
        st.info("👈 Fill in the patient details in the sidebar and click **Predict Diabetes Risk**")
        st.markdown("""
        **About This Application:**
        - Trained on 100,000 patient records
        - Uses XGBoost with hyperparameter tuning
        - Achieves ~97% ROC-AUC on validation data
        - Provides probability score and risk category
        - ⚠️ For educational purposes only — not a substitute for medical advice
        """)

# ── Footer ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🩺 Diabetes Prediction System &nbsp;|&nbsp; Built with Streamlit + XGBoost &nbsp;|&nbsp;
    ⚠️ <em>For educational/research purposes only. Not a substitute for professional medical advice.</em>
</div>
""", unsafe_allow_html=True)