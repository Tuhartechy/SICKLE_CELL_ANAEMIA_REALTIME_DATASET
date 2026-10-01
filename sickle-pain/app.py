"""
Sickle Cell Disease: Pain Severity Predictor & Clinical Scaling Interface
Modern, Minimalist Clinical UI built with Streamlit.
"""

import os
import json
import joblib
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Sickle Cell Pain Severity Predictor",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

# --- MODERN MINIMALIST STYLING ---
st.markdown("""
<style>
    /* Clean modern typography and background */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #0f172a;
    }

    /* Minimalist Top Navigation Banner */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 20px 24px;
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
    }
    .app-title {
        font-size: 22px;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .app-subtitle {
        font-size: 13px;
        color: #94a3b8;
        margin-top: 4px;
    }
    .status-badge {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        color: #38bdf8;
    }

    /* Clean Card Containers */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 14px;
        transition: all 0.2s ease-in-out;
    }
    .metric-card:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    }
    .card-heading {
        font-size: 14px;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Triage Result Cards */
    .triage-mild {
        background: #f0fdf4;
        border: 1.5px solid #22c55e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: #14532d;
    }
    .triage-moderate {
        background: #fffbeb;
        border: 1.5px solid #f59e0b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: #78350f;
    }
    .triage-severe {
        background: #fef2f2;
        border: 1.5px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: #7f1d1d;
    }

    /* Minimal Pill Badges */
    .pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
    }
    .pill-red { background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }
    .pill-amber { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .pill-green { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }

    /* Button Enhancements */
    div.stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_metadata():
    """Loads the serialized model and metadata."""
    model_path = os.path.join(MODELS_DIR, 'best_model.joblib')
    meta_path = os.path.join(MODELS_DIR, 'metadata.json')
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None, None
        
    model = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    return model, metadata


@st.cache_data
def load_comparison_data():
    """Loads model evaluation metrics."""
    comp_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    if os.path.exists(comp_path):
        return pd.read_csv(comp_path)
    return None


def init_session_state(metadata):
    """Initializes session state with default median values."""
    defaults = metadata.get('feature_defaults', {})
    if 'p_age' not in st.session_state:
        st.session_state['p_age'] = int(defaults.get('Age', 25))
        st.session_state['p_sex'] = defaults.get('Sex', 'F')
        st.session_state['p_geno'] = defaults.get('Genotype', 'SS')
        st.session_state['p_hr'] = int(defaults.get('Heart_Rate', 90))
        st.session_state['p_rr'] = int(defaults.get('Respiratory_Rate', 20))
        st.session_state['p_spo2'] = float(defaults.get('SpO2', 98.0))
        st.session_state['p_temp'] = float(defaults.get('Temperature', 98.6))
        st.session_state['p_bps'] = int(defaults.get('BP_Systolic', 110))
        st.session_state['p_bpd'] = int(defaults.get('BP_Diastolic', 70))
        st.session_state['p_hb'] = float(defaults.get('Hemoglobin', 8.5))
        st.session_state['p_wbc'] = float(defaults.get('WBC', 9000.0))
        st.session_state['p_plt'] = float(defaults.get('Platelets', 250000.0))
        st.session_state['p_rdw'] = float(defaults.get('RDW', 20.0))
        st.session_state['p_bili'] = float(defaults.get('Bilirubin_Total', 2.0))
        st.session_state['p_creat'] = float(defaults.get('Creatinine', 0.55))


def apply_preset(preset_type):
    """Loads clinical patient scenario presets."""
    if preset_type == 'severe':
        st.session_state['p_age'] = 28
        st.session_state['p_sex'] = 'F'
        st.session_state['p_geno'] = 'SS'
        st.session_state['p_hr'] = 126
        st.session_state['p_rr'] = 28
        st.session_state['p_spo2'] = 93.0
        st.session_state['p_temp'] = 101.4
        st.session_state['p_bps'] = 105
        st.session_state['p_bpd'] = 65
        st.session_state['p_hb'] = 5.1
        st.session_state['p_wbc'] = 18900.0
        st.session_state['p_plt'] = 85000.0
        st.session_state['p_rdw'] = 25.5
        st.session_state['p_bili'] = 6.8
        st.session_state['p_creat'] = 1.10
    elif preset_type == 'moderate':
        st.session_state['p_age'] = 34
        st.session_state['p_sex'] = 'M'
        st.session_state['p_geno'] = 'SS'
        st.session_state['p_hr'] = 102
        st.session_state['p_rr'] = 22
        st.session_state['p_spo2'] = 97.0
        st.session_state['p_temp'] = 99.2
        st.session_state['p_bps'] = 115
        st.session_state['p_bpd'] = 72
        st.session_state['p_hb'] = 7.6
        st.session_state['p_wbc'] = 11200.0
        st.session_state['p_plt'] = 180000.0
        st.session_state['p_rdw'] = 21.0
        st.session_state['p_bili'] = 3.4
        st.session_state['p_creat'] = 0.70
    elif preset_type == 'mild':
        st.session_state['p_age'] = 42
        st.session_state['p_sex'] = 'F'
        st.session_state['p_geno'] = 'AS'
        st.session_state['p_hr'] = 74
        st.session_state['p_rr'] = 16
        st.session_state['p_spo2'] = 99.0
        st.session_state['p_temp'] = 98.6
        st.session_state['p_bps'] = 120
        st.session_state['p_bpd'] = 75
        st.session_state['p_hb'] = 12.6
        st.session_state['p_wbc'] = 5800.0
        st.session_state['p_plt'] = 295000.0
        st.session_state['p_rdw'] = 13.8
        st.session_state['p_bili'] = 0.8
        st.session_state['p_creat'] = 0.60


def main():
    model, metadata = load_model_and_metadata()
    comparison_df = load_comparison_data()

    if model is None or metadata is None:
        st.error("⚠️ Model components not detected. Please ensure training scripts have completed.")
        return

    init_session_state(metadata)

    # --- TOP HEADER ---
    st.markdown("""
    <div class="app-header">
        <div>
            <div class="app-title">🩸 Sickle Cell Pain Severity Predictor</div>
            <div class="app-subtitle">Objective Clinical Triage & Crisis Severity Classification</div>
        </div>
        <div class="status-badge">● Stacking Ensemble Ready</div>
    </div>
    """, unsafe_allow_html=True)

    # --- SIDEBAR: CLINICAL REFERENCE & SCENARIOS ---
    with st.sidebar:
        st.markdown("### ⚡ Quick Scenarios")
        st.caption("One-click presets for typical clinical presentations:")

        if st.button("🔴 Severe Crisis Profile", use_container_width=True):
            apply_preset('severe')
            st.rerun()

        if st.button("🟠 Moderate Pain Profile", use_container_width=True):
            apply_preset('moderate')
            st.rerun()

        if st.button("🟢 Stable / Baseline Profile", use_container_width=True):
            apply_preset('mild')
            st.rerun()

        if st.button("🔄 Reset to Normal Medians", use_container_width=True):
            defaults = metadata.get('feature_defaults', {})
            st.session_state['p_age'] = int(defaults.get('Age', 25))
            st.session_state['p_sex'] = defaults.get('Sex', 'F')
            st.session_state['p_geno'] = defaults.get('Genotype', 'SS')
            st.session_state['p_hr'] = int(defaults.get('Heart_Rate', 90))
            st.session_state['p_rr'] = int(defaults.get('Respiratory_Rate', 20))
            st.session_state['p_spo2'] = float(defaults.get('SpO2', 98.0))
            st.session_state['p_temp'] = float(defaults.get('Temperature', 98.6))
            st.session_state['p_bps'] = int(defaults.get('BP_Systolic', 110))
            st.session_state['p_bpd'] = int(defaults.get('BP_Diastolic', 70))
            st.session_state['p_hb'] = float(defaults.get('Hemoglobin', 8.5))
            st.session_state['p_wbc'] = float(defaults.get('WBC', 9000.0))
            st.session_state['p_plt'] = float(defaults.get('Platelets', 250000.0))
            st.session_state['p_rdw'] = float(defaults.get('RDW', 20.0))
            st.session_state['p_bili'] = float(defaults.get('Bilirubin_Total', 2.0))
            st.session_state['p_creat'] = float(defaults.get('Creatinine', 0.55))
            st.rerun()

        st.divider()
        st.markdown("### 🔬 Pathophysiology")
        circ_img = os.path.join(ASSETS_DIR, 'sickle_circulation.jpg')
        if os.path.exists(circ_img):
            st.image(circ_img, caption="Microvascular Vaso-Occlusion Mechanism", use_container_width=True)

        st.caption(
            "Rigid, sickled RBCs adhere to vascular endothelium, obstructing capillary blood flow and causing tissue ischemia."
        )

    # --- MAIN TABS ---
    tab_assessment, tab_scale, tab_analytics = st.tabs([
        "🩺 Patient Assessment",
        "📐 Pain Scaling & Pathology",
        "📊 Model Benchmarks & Metrics"
    ])

    # -------------------------------------------------------------
    # TAB 1: PATIENT ASSESSMENT
    # -------------------------------------------------------------
    with tab_assessment:
        # Visual Pain Scale Mini-Banner
        pain_scale_img = os.path.join(ASSETS_DIR, 'sickle_pain_scale.jpg')
        with st.expander("📐 View Clinical Pain Severity Scale (Visual Reference)", expanded=True):
            if os.path.exists(pain_scale_img):
                st.image(
                    pain_scale_img, 
                    caption="Standardized Pain Severity Scale for Sickle Cell Disease (Score 0-3 Mild, 4-6 Moderate, 7-10 Severe)", 
                    use_container_width=True
                )

        # Real-time Physiological Warning Chips
        pills = []
        if st.session_state['p_temp'] >= 100.4:
            pills.append('<span class="pill pill-red">🌡️ Fever ≥100.4°F</span>')
        if st.session_state['p_spo2'] < 95.0:
            pills.append('<span class="pill pill-red">🫁 Hypoxia SpO2 &lt;95%</span>')
        if st.session_state['p_hr'] >= 105:
            pills.append('<span class="pill pill-amber">💓 Tachycardia ≥105 bpm</span>')
        if st.session_state['p_hb'] <= 6.0:
            pills.append('<span class="pill pill-red">🩸 Severe Anemia Hb ≤6.0</span>')
        if st.session_state['p_wbc'] >= 13000.0:
            pills.append('<span class="pill pill-amber">🛡️ Leukocytosis WBC ≥13k</span>')

        if pills:
            st.markdown("".join(pills), unsafe_allow_html=True)
            st.write("")

        # Clinical Assessment Form
        with st.form("assessment_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**1. Demographics & Genotype**")
                age = st.number_input("Age (years)", min_value=1, max_value=100, value=int(st.session_state['p_age']))
                sex = st.selectbox("Biological Sex", options=['F', 'M'], index=0 if st.session_state['p_sex'] == 'F' else 1)
                
                geno_options = ['SS', 'AS', 'SF', 'SBetaThal', 'Negative', 'Unknown']
                cur_geno = st.session_state['p_geno']
                g_idx = geno_options.index(cur_geno) if cur_geno in geno_options else 0
                genotype = st.selectbox("Hemoglobin Genotype", options=geno_options, index=g_idx)
                st.caption("SS: Sickle Anemia | AS: Trait | SBetaThal: β-Thalassemia")

            with c2:
                st.markdown("**2. Vital Signs**")
                heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=220, value=int(st.session_state['p_hr']))
                resp_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=8, max_value=60, value=int(st.session_state['p_rr']))
                spo2 = st.number_input("SpO2 (%)", min_value=50.0, max_value=100.0, value=float(st.session_state['p_spo2']), step=0.5)
                temp = st.number_input("Temperature (°F)", min_value=90.0, max_value=110.0, value=float(st.session_state['p_temp']), step=0.1)
                bp_sys = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=240, value=int(st.session_state['p_bps']))
                bp_dia = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=140, value=int(st.session_state['p_bpd']))

            with c3:
                st.markdown("**3. Laboratory Panel**")
                sim_missing = st.checkbox("Laboratory Results Pending / Blank", value=False)
                
                if not sim_missing:
                    hb = st.number_input("Hemoglobin (g/dL)", min_value=1.0, max_value=25.0, value=float(st.session_state['p_hb']), step=0.1)
                    wbc = st.number_input("WBC Count (/mcL)", min_value=1000.0, max_value=60000.0, value=float(st.session_state['p_wbc']), step=100.0)
                    plt_val = st.number_input("Platelets (/mcL)", min_value=10000.0, max_value=1000000.0, value=float(st.session_state['p_plt']), step=5000.0)
                    rdw = st.number_input("RDW (%)", min_value=5.0, max_value=45.0, value=float(st.session_state['p_rdw']), step=0.1)
                    bili = st.number_input("Total Bilirubin (mg/dL)", min_value=0.1, max_value=35.0, value=float(st.session_state['p_bili']), step=0.1)
                    creat = st.number_input("Serum Creatinine (mg/dL)", min_value=0.1, max_value=15.0, value=float(st.session_state['p_creat']), step=0.05)
                else:
                    st.info("⚡ Automatic median imputation enabled for pending labs.")
                    hb, wbc, plt_val, rdw, bili, creat = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

            st.write("")
            calc_btn = st.form_submit_button("🔍 Calculate Pain Severity & Triage", use_container_width=True)

        # Output Section
        if calc_btn:
            with st.spinner("Processing clinical parameters via Stacking Ensemble..."):
                time.sleep(0.3)
                
                input_df = pd.DataFrame([{
                    'Age': age,
                    'Heart_Rate': heart_rate,
                    'Respiratory_Rate': resp_rate,
                    'SpO2': spo2,
                    'Temperature': temp,
                    'BP_Systolic': bp_sys,
                    'BP_Diastolic': bp_dia,
                    'Hemoglobin': hb,
                    'WBC': wbc,
                    'Platelets': plt_val,
                    'RDW': rdw,
                    'Bilirubin_Total': bili,
                    'Creatinine': creat,
                    'Sex': sex,
                    'Genotype': genotype
                }])

                try:
                    pred_idx = model.predict(input_df)[0]
                    probas = model.predict_proba(input_df)[0]
                    classes = metadata['class_names']
                    label = classes[pred_idx]

                    st.write("")
                    st.subheader("📋 Predicted Severity & Action Plan")

                    r_col1, r_col2 = st.columns([1.1, 1.3])

                    with r_col1:
                        if label == 'Mild':
                            st.markdown("""
                            <div class="triage-mild">
                                <h2 style="margin: 0; font-size: 24px;">🟢 MILD PAIN</h2>
                                <p style="font-weight: 600; margin: 4px 0 10px 0;">Pain Scale Equivalent: 0 – 3</p>
                                <p style="font-size: 13px; margin: 0;">Physiological signs are within baseline limits. Mild discomfort manageable via outpatient protocol.</p>
                                <hr style="border: 0; border-top: 1px solid rgba(0,0,0,0.1); margin: 10px 0;">
                                <div style="font-size: 12px; text-align: left;">
                                    <strong>Clinical Guidance:</strong> Oral hydration, standard oral analgesics (paracetamol / NSAIDs), and monitoring.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif label == 'Moderate':
                            st.markdown("""
                            <div class="triage-moderate">
                                <h2 style="margin: 0; font-size: 24px;">🟠 MODERATE PAIN</h2>
                                <p style="font-weight: 600; margin: 4px 0 10px 0;">Pain Scale Equivalent: 4 – 6</p>
                                <p style="font-size: 13px; margin: 0;">Sub-acute physiological distress and elevated pain markers detected.</p>
                                <hr style="border: 0; border-top: 1px solid rgba(0,0,0,0.1); margin: 10px 0;">
                                <div style="font-size: 12px; text-align: left;">
                                    <strong>Clinical Guidance:</strong> Emergency observation, aggressive oral/IV hydration, oral opioid protocol, repeat vitals in 60 mins.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="triage-severe">
                                <h2 style="margin: 0; font-size: 24px;">🔴 SEVERE PAIN</h2>
                                <p style="font-weight: 600; margin: 4px 0 10px 0;">Pain Scale Equivalent: 7 – 10</p>
                                <p style="font-size: 13px; margin: 0;">Acute vaso-occlusive crisis indicators present with compensatory tachycardia / tachypnea.</p>
                                <hr style="border: 0; border-top: 1px solid rgba(0,0,0,0.1); margin: 10px 0;">
                                <div style="font-size: 12px; text-align: left;">
                                    <strong>Clinical Guidance:</strong> Immediate inpatient triage, parenteral opioid analgesia within 30 min, IV fluid protocol, O2 therapy.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                    with r_col2:
                        st.markdown("##### Ensemble Probability Distribution")
                        p_dict = {c: p for c, p in zip(classes, probas)}
                        
                        for c in ['Mild', 'Moderate', 'Severe']:
                            p_val = p_dict.get(c, 0.0)
                            color_icon = "🟢" if c == 'Mild' else ("🟠" if c == 'Moderate' else "🔴")
                            st.write(f"**{color_icon} {c}:** `{p_val * 100:.1f}%`")
                            st.progress(float(p_val))

                        st.write("")
                        st.caption("Prediction generated by Stacking Ensemble (RF + XGBoost + SVM + LR + KNN).")

                except Exception as err:
                    st.error(f"Inference Error: {err}")

    # -------------------------------------------------------------
    # TAB 2: PAIN SCALING & PATHOLOGY
    # -------------------------------------------------------------
    with tab_scale:
        st.subheader("Sickle Cell Pain Severity Scaling & Pathology")
        st.write(
            "In Sickle Cell Disease, pain episodes occur when red blood cells become sickle-shaped, rigid, and clump together in small blood vessels, "
            "blocking blood and oxygen from reaching tissues."
        )

        col_s1, col_s2 = st.columns(2)

        with col_s1:
            st.markdown("#### 1. Standardized Clinical Pain Scale")
            if os.path.exists(pain_scale_img):
                st.image(
                    pain_scale_img, 
                    caption="Figure 1: Sickle Cell Pain Severity Infographic & Triage Scale", 
                    use_container_width=True
                )
            st.markdown("""
            - **Mild (Score 0 – 3)**: Manageable pain, usually managed at home with hydration and oral analgesics.
            - **Moderate (Score 4 – 6)**: Escalating pain requiring medical assessment, close observation, and analgesia titration.
            - **Severe (Score 7 – 10)**: Excruciating pain; clinical emergency necessitating hospitalization, rapid IV analgesia, and hydration.
            """)

        with col_s2:
            st.markdown("#### 2. Microvascular Vaso-Occlusion")
            if os.path.exists(circ_img):
                st.image(
                    circ_img, 
                    caption="Figure 2: Normal Biconcave RBCs vs. Sickled RBC Vaso-Occlusion", 
                    use_container_width=True
                )
            st.markdown("""
            - **Normal Flow**: Flexible biconcave erythrocytes slide smoothly through microcapillaries without obstruction.
            - **Vaso-Occlusion**: Deoxygenated HbS polymerizes, transforming cells into rigid crescents that adhere to vessel walls, creating local ischemia and inflammatory cascades.
            """)

    # -------------------------------------------------------------
    # TAB 3: MODEL BENCHMARKS & METRICS
    # -------------------------------------------------------------
    with tab_analytics:
        st.subheader("Model Benchmarks & Cross-Validation Results")
        st.write(
            "Evaluation carried out using **Repeated Stratified 5-Fold Cross-Validation (25 rounds)** "
            "and **Leave-One-Out Cross-Validation (LOOCV)** on the pilot patient dataset ($N=23$). "
            "All pipelines evaluate identical training and test splits with strict zero-leakage preprocessing."
        )

        if comparison_df is not None:
            st.markdown("##### Performance Benchmark Table")
            st.dataframe(
                comparison_df.style.format({
                    'CV_Macro_F1_Mean': '{:.4f}',
                    'CV_Macro_F1_Std': '{:.4f}',
                    'CV_Bal_Acc_Mean': '{:.4f}',
                    'CV_Accuracy_Mean': '{:.4f}',
                    'LOOCV_Macro_F1': '{:.4f}',
                    'LOOCV_Accuracy': '{:.4f}'
                }).highlight_max(subset=['CV_Macro_F1_Mean', 'CV_Bal_Acc_Mean', 'CV_Accuracy_Mean'], color='#dcfce7'),
                use_container_width=True
            )

        st.write("")
        st.markdown("##### Benchmark Visualizations")
        bc1, bc2 = st.columns(2)

        with bc1:
            f1_path = os.path.join(RESULTS_DIR, 'macro_f1_comparison.png')
            if os.path.exists(f1_path):
                st.image(f1_path, caption="Cross-Validation Macro F1 Score (Mean ± Std)", use_container_width=True)

        with bc2:
            cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix_stacking_ensemble.png')
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Confusion Matrix: Stacking Ensemble (Out-of-Fold)", use_container_width=True)

        st.write("")
        fi_path = os.path.join(RESULTS_DIR, 'feature_importance.png')
        if os.path.exists(fi_path):
            st.image(fi_path, caption="Feature Importance: Random Forest Gini vs. Permutation Importance", use_container_width=True)

        st.info(
            "**Key Benchmark Finding:** The Stacking Ensemble achieved the highest nominal Macro F1 (0.2173), "
            "slightly ahead of SVM (0.1956). Systemic indicators of physiological stress (Respiratory Rate, WBC, and Hemoglobin) "
            "emerged as the strongest non-invasive predictors of pain severity."
        )


if __name__ == '__main__':
    main()
