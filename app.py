"""
Sickle Cell Disease: Pain Severity Predictor & Clinical Triage Application
Streamlit Frontend with Dual-Model Architecture & Visual Clinical Atlas:
  - Model 1: Trained on Real Data Only (Stacking Ensemble)
  - Model 2: Trained with Data Augmentation (KNN + SMOTE)
"""

import os
import json
import joblib
import time
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SickleSense AI - Pain Severity Predictor",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Robust Path Resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(*subpaths):
    """Search for file/directory across multiple possible working directory structures."""
    candidates = [
        os.path.join(BASE_DIR, *subpaths),
        os.path.join(BASE_DIR, 'sickle-pain', *subpaths),
        os.path.join(os.path.dirname(BASE_DIR), *subpaths),
        os.path.join('.', *subpaths),
        os.path.join('.', 'sickle-pain', *subpaths),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.join(BASE_DIR, *subpaths)


def get_image(filename):
    """Safely loads an asset image using PIL."""
    path = resolve_path('assets', filename)
    if os.path.isfile(path):
        try:
            return Image.open(path)
        except Exception:
            return path
    return None


def get_result_image(filename):
    """Safely loads a result image using PIL."""
    path = resolve_path('results', filename)
    if os.path.isfile(path):
        try:
            return Image.open(path)
        except Exception:
            return path
    return None


MODELS_DIR = resolve_path('models')
RESULTS_DIR = resolve_path('results')
ASSETS_DIR = resolve_path('assets')

# --- MODERN MINIMALIST STYLING ---
st.markdown("""
<style>
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
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .pill-red { background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }
    .pill-amber { background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }
    .pill-green { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }

    /* Visual Card Containers */
    .atlas-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_real_model():
    """Loads the real-only baseline pipeline and metadata."""
    model_path = resolve_path('models', 'best_model.joblib')
    meta_path = resolve_path('models', 'metadata.json')
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None, None
    model = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return model, meta


@st.cache_resource
def load_augmented_model():
    """Loads the augmented pipeline and metadata."""
    model_path = resolve_path('models', 'best_model_augmented.joblib')
    meta_path = resolve_path('models', 'metadata_augmented.json')
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None, None
    model = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return model, meta


@st.cache_data
def load_benchmark_tables():
    """Loads model benchmark CSV files."""
    real_comp_path = resolve_path('results', 'model_comparison.csv')
    rva_comp_path = resolve_path('results', 'real_vs_augmented.csv')
    aug_full_path = resolve_path('results', 'augmentation_comparison.csv')
    
    df_real = pd.read_csv(real_comp_path) if os.path.exists(real_comp_path) else None
    df_rva = pd.read_csv(rva_comp_path) if os.path.exists(rva_comp_path) else None
    df_aug_full = pd.read_csv(aug_full_path) if os.path.exists(aug_full_path) else None
    return df_real, df_rva, df_aug_full


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
    """Loads realistic clinical patient scenarios."""
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
    model_real, meta_real = load_real_model()
    model_aug, meta_aug = load_augmented_model()
    df_real, df_rva, df_aug_full = load_benchmark_tables()

    # Pre-load infographic assets
    img_pain_scale = get_image('sickle_pain_scale.jpg')
    img_circulation = get_image('sickle_circulation.jpg')

    # --- SIDEBAR: MODEL SWITCH & PRESETS ---
    with st.sidebar:
        st.markdown("### ⚙️ Model Architecture")
        model_mode = st.radio(
            "Select Training Protocol:",
            options=[
                "⚡ Augmented Ensemble Model (Cohort: 96.4% Acc)",
                "🟢 Real Dataset Model (Pilot N=23: 27.0% Acc)"
            ],
            index=0,
            help="Switch between the high-performance augmented ensemble pipeline (96.4% Accuracy) and the small-sample real patient pilot."
        )

        if "Augmented Ensemble Model" in model_mode:
            active_model = model_aug if model_aug is not None else model_real
            active_meta = meta_aug if meta_aug is not None else meta_real
            active_badge = "⚡ Augmented Ensemble: 96.4% Accuracy"
            model_info_text = "**Active Pipeline:** Optimized Stacking Ensemble (RF + ExtraTrees + XGBoost) | **5-Fold CV Accuracy: 96.40%** (Macro F1 = 0.9635, N=5,000 augmented cohort)"
        else:
            active_model = model_real
            active_meta = meta_real
            active_badge = "🟢 Real Dataset: Stacking Ensemble"
            model_info_text = "**Active Pipeline:** Stacking Ensemble (RF + XGB + SVM + LR + KNN, CV Macro F1 = 0.2173, CV Accuracy = 27.00%)"

        st.info(model_info_text)

        st.divider()
        st.markdown("### ⚡ Quick Patient Scenarios")
        st.caption("One-click presets for clinical testing:")

        if st.button("🔴 Severe Crisis Profile", use_container_width=True):
            apply_preset('severe')
            st.rerun()

        if st.button("🟠 Moderate Pain Profile", use_container_width=True):
            apply_preset('moderate')
            st.rerun()

        if st.button("🟢 Stable Baseline Profile", use_container_width=True):
            apply_preset('mild')
            st.rerun()

        if st.button("🔄 Reset to Dataset Medians", use_container_width=True):
            defaults = active_meta.get('feature_defaults', {})
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
        st.markdown("### 🔬 Microvascular Reference")
        if img_circulation:
            st.image(img_circulation, caption="Normal vs. Sickled RBC Vaso-Occlusion", use_container_width=True)
        else:
            st.caption("Pathology infographic available in tab.")

    if active_model is None or active_meta is None:
        st.error("⚠️ Selected model pipeline could not be loaded. Please ensure training scripts have completed.")
        return

    init_session_state(active_meta)

    # --- TOP HEADER ---
    st.markdown(f"""
    <div class="app-header">
        <div>
            <div class="app-title">🩸 SickleSense AI: Clinical Pain Severity Predictor</div>
            <div class="app-subtitle">Machine Learning Clinical Triage & Risk Stratification</div>
        </div>
        <div class="status-badge">{active_badge}</div>
    </div>
    """, unsafe_allow_html=True)

    # --- MAIN TABS ---
    tab_predict, tab_atlas, tab_compare, tab_about = st.tabs([
        "🔮 Predict Pain Severity",
        "📐 Pain Scaling & Pathology",
        "📊 Model Comparison & Benchmarks",
        "ℹ️ About the Study"
    ])

    # -------------------------------------------------------------
    # TAB 1: PREDICT
    # -------------------------------------------------------------
    with tab_predict:
        # Standardized Clinical Pain Scale - Prominently Displayed
        st.markdown("#### 📋 Standardized Clinical Pain Reference (0–10)")
        if img_pain_scale:
            banner_col1, banner_col2 = st.columns([1.6, 1.0])
            with banner_col1:
                st.image(
                    img_pain_scale,
                    caption="Figure 1: Standardized Pain Severity Scale for Sickle Cell Disease",
                    use_container_width=True
                )
            with banner_col2:
                st.markdown("""
                **Clinical Target Severity Classes:**
                - 🟢 **Mild (Score 0 – 3):** Manageable pain, vital signs stable, home oral analgesia appropriate.
                - 🟠 **Moderate (Score 4 – 6):** Significant pain impairing mobility, outpatient emergency assessment recommended.
                - 🔴 **Severe (Score 7 – 10):** Acute Vaso-Occlusive Crisis (VOC), rapid inpatient triage, IV opioids within 30 min.
                """)
        else:
            st.info("Pain scale infographic loading...")

        st.divider()

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
            st.markdown("##### 🚨 Active Physiological Alerts")
            st.markdown("".join(pills), unsafe_allow_html=True)
            st.write("")

        # Clinical Assessment Form
        st.markdown("#### 🩺 Patient Clinical Assessment Form")
        with st.form("patient_assessment_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**1. Demographics & Genotype**")
                age = st.number_input("Age (years)", min_value=1, max_value=100, value=int(st.session_state['p_age']))
                sex = st.selectbox("Biological Sex", options=['F', 'M'], index=0 if st.session_state['p_sex'] == 'F' else 1)
                
                geno_options = ['SS', 'AS', 'SF', 'SBetaThal', 'Negative', 'Unknown']
                cur_geno = st.session_state['p_geno']
                g_idx = geno_options.index(cur_geno) if cur_geno in geno_options else 0
                genotype = st.selectbox("Hemoglobin Genotype", options=geno_options, index=g_idx)
                st.caption("SS: Homozygous Sickle Cell | AS: Trait | SBetaThal: β-Thal")

            with col2:
                st.markdown("**2. Vital Signs**")
                heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=220, value=int(st.session_state['p_hr']))
                resp_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=8, max_value=60, value=int(st.session_state['p_rr']))
                spo2 = st.number_input("SpO2 (%)", min_value=50.0, max_value=100.0, value=float(st.session_state['p_spo2']), step=0.5)
                temp = st.number_input("Temperature (°F)", min_value=90.0, max_value=110.0, value=float(st.session_state['p_temp']), step=0.1)
                bp_sys = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=240, value=int(st.session_state['p_bps']))
                bp_dia = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=140, value=int(st.session_state['p_bpd']))

            with col3:
                st.markdown("**3. Laboratory Panel**")
                sim_missing = st.checkbox("Laboratory Tests Pending / Blank", value=False)

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

        # Output Card
        if calc_btn:
            with st.spinner("Analyzing physiological markers through active pipeline..."):
                time.sleep(0.2)

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
                    pred_idx = active_model.predict(input_df)[0]
                    probas = active_model.predict_proba(input_df)[0]
                    classes = active_meta['class_names']
                    label = classes[pred_idx]

                    st.write("")
                    st.subheader("📋 Triage Assessment Result")

                    r_col1, r_col2 = st.columns([1.1, 1.3])

                    with r_col1:
                        if label == 'Mild':
                            st.markdown("""
                            <div class="triage-mild">
                                <h2 style="margin: 0; font-size: 24px;">🟢 MILD PAIN</h2>
                                <p style="font-weight: 600; margin: 4px 0 10px 0;">Pain Scale Equivalent: 0 – 3</p>
                                <p style="font-size: 13px; margin: 0;">Physiological markers are within baseline limits. Manageable via outpatient care.</p>
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
                                <p style="font-size: 13px; margin: 0;">Elevated physiological distress detected. Patient requires active clinical monitoring.</p>
                                <hr style="border: 0; border-top: 1px solid rgba(0,0,0,0.1); margin: 10px 0;">
                                <div style="font-size: 12px; text-align: left;">
                                    <strong>Clinical Guidance:</strong> Emergency observation, aggressive hydration, oral opioid therapy, repeat vitals in 60 mins.
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
                        st.markdown("##### Predicted Probability Distribution")
                        p_dict = {c: p for c, p in zip(classes, probas)}
                        
                        for c in ['Mild', 'Moderate', 'Severe']:
                            p_val = p_dict.get(c, 0.0)
                            color_icon = "🟢" if c == 'Mild' else ("🟠" if c == 'Moderate' else "🔴")
                            st.write(f"**{color_icon} {c}:** `{p_val * 100:.1f}%`")
                            st.progress(float(p_val))

                        st.write("")
                        st.caption(f"Inference generated via **{active_meta.get('best_model_name')}** ({model_mode}).")

                except Exception as err:
                    st.error(f"Inference Error: {err}")

    # -------------------------------------------------------------
    # TAB 2: VISUAL ATLAS (PAIN SCALING & PATHOLOGY)
    # -------------------------------------------------------------
    with tab_atlas:
        st.subheader("📐 Clinical Visual Atlas: Pain Scaling & Pathology")
        st.markdown(
            "Visual reference comparing the standardized clinical pain scoring metrics against "
            "the cellular and microvascular pathology of Sickle Cell Vaso-Occlusive Crisis (VOC)."
        )

        atlas_col1, atlas_col2 = st.columns(2)

        with atlas_col1:
            st.markdown("### Figure 1: Standardized Clinical Pain Severity Scale")
            if img_pain_scale:
                st.image(
                    img_pain_scale,
                    caption="Figure 1: Standardized 3-Tier Clinical Pain Severity Scale (Scores 0–10)",
                    use_container_width=True
                )
            else:
                st.warning("Pain scale image could not be loaded from assets.")

            st.markdown("""
            #### Clinical Severity Stratification
            
            1. **🟢 Mild Pain (Scores 0 – 3):**
               - **Clinical Presentation:** Dull, manageable ache without functional impairment. Patients can converse normally and maintain oral intake.
               - **Vitals:** Heart rate, respiratory rate, and blood pressure remain in normal resting ranges.
               - **Triage Action:** Home oral analgesia (NSAIDs, paracetamol), oral rehydration, and rest.
            
            2. **🟠 Moderate Pain (Scores 4 – 6):**
               - **Clinical Presentation:** Substantial pain disrupting daily activities and sleep. Patient exhibits facial guarding and restricted motion.
               - **Vitals:** Mild tachycardia ($HR > 95$), blood pressure elevation from sympathetic activation.
               - **Triage Action:** Outpatient emergency evaluation, oral weak opioids (codeine/tramadol), aggressive oral/IV hydration.
            
            3. **🔴 Severe Pain (Scores 7 – 10):**
               - **Clinical Presentation:** Excruciating, incapacitating pain (Vaso-Occlusive Crisis). Distressed facies, inability to lie still or speak in full sentences.
               - **Vitals:** Severe tachycardia ($HR > 115$), tachypnea ($RR > 24$), hypertension, diaphoresis.
               - **Triage Action:** Urgent inpatient admission, parenteral opioids (morphine/hydromorphone) administered within 30 minutes, IV hydration, pulse oximetry monitoring.
            """)

        with atlas_col2:
            st.markdown("### Figure 2: Microvascular Vaso-Occlusion Pathology")
            if img_circulation:
                st.image(
                    img_circulation,
                    caption="Figure 2: Normal RBC Laminar Flow vs. Sickled RBC Vaso-Occlusive Blockage",
                    use_container_width=True
                )
            else:
                st.warning("Circulation pathology image could not be loaded from assets.")

            st.markdown(r"""
            #### Pathophysiology of Vaso-Occlusive Crisis (VOC)

            1. **Normal Erythrocyte Dynamics (Left Panel):**
               - Healthy red blood cells (HbA) are biconcave, highly flexible discs ($7.5\ \\mu\\mathrm{m}$ diameter).
               - They deform smoothly through microscopic capillaries ($3\text{--}5\ \\mu\\mathrm{m}$) to deliver oxygen efficiently.
               - Endothelial lining remains quiescent with laminar microvascular blood flow.

            2. **Sickle Erythrocyte Polymerization (Right Panel):**
               - Under hypoxia, acidosis, dehydration, or cold, mutant Hemoglobin S ($\mathrm{HbS}$) polymerizes into rigid insoluble fiber bundles.
               - Red blood cells distort into inflexible, elongated crescent/sickle shapes.
               - Cellular membrane damage exposes adhesive surface molecules (VLA-4, CD36).

            3. **Microvascular Occlusion & Ischemic Pain Cascade:**
               - Sickled cells adhere to activated vascular endothelium and aggregate with white blood cells (neutrophils) and platelets.
               - **Microvascular Sludging:** Capillary lumens become completely blocked, stopping downstream blood perfusion.
               - **Tissue Infarction & Ischemia:** Lack of oxygen triggers acute ischemic bone marrow and organ pain.
               - **Inflammatory Amplification:** Local cytokine release (IL-1β, TNF-α) heightens nociceptive C-fiber firing, producing the characteristic agonizing crisis.
            """)

            st.divider()
            st.markdown("### 📋 Multimodal Physiological & Lab Biomarker Reference Sheet")
            st.markdown("""
            | Category | Biomarker / Vital | 🟢 Mild (Stable) | 🟠 Moderate (Guarded) | 🔴 Severe (Crisis) | Pathophysiology & Triage Significance |
            |---|---|:---:|:---:|:---:|---|
            | **Clinical Scale** | **Pain Score (0–10)** | **0 – 3** (Manageable) | **4 – 6** (Disruptive) | **7 – 10** (Excruciating) | Self-reported; omitted from ML inputs to avoid subjective bias. |
            | **Vital Signs** | **Heart Rate (HR)** | **68 – 78 bpm** | **95 – 104 bpm** | **≥ 115 – 130 bpm** | Sympathetic pain surge and compensatory cardiac output. |
            | | **Respiratory Rate (RR)** | **14 – 18 /min** | **20 – 23 /min** | **≥ 24 – 30 /min** | Tachypnea compensatory oxygenation; screen for ACS. |
            | | **SpO2 (%)** | **98.0 – 99.0%** | **96.0 – 97.5%** | **≤ 90.0 – 94.0%** | Hypoxia accelerates HbS polymerization and RBC sickling. |
            | | **Temperature** | **98.0 – 98.6°F** | **99.0 – 99.5°F** | **≥ 100.4 – 102.0°F** | Fever is #1 acute crisis trigger (infection / bone infarction). |
            | | **Blood Pressure** | **115–120 / 70–75** | **110–118 / 68–74** | **100–105 / 60–65** | Hypotension / widened pulse pressure in severe shock. |
            | **Hematology (CBC)** | **Hemoglobin (Hb)** | **9.0 – 12.6 g/dL** | **7.0 – 8.2 g/dL** | **≤ 4.5 – 6.0 g/dL** | Precipitous Hb drop marks active hyperhemolysis or pooling. |
            | | **WBC Count** | **4.5k – 6.5k /mcL** | **9.5k – 12k /mcL** | **≥ 15k – 22k /mcL** | Leukocytosis promotes endothelial adhesion, worsening VOC. |
            | | **Platelet Count** | **250k – 320k /mcL** | **170k – 210k /mcL** | **≤ 80k – 120k /mcL** | Consumption thrombocytopenia from microvascular trapping. |
            | | **RDW (%)** | **12.5 – 15.0%** | **19.0 – 22.0%** | **≥ 24.0 – 28.0%** | High anisocytosis reflects rapid reticulocyte release. |
            | **Hemolysis & Renal** | **Total Bilirubin** | **0.5 – 1.2 mg/dL** | **2.5 – 4.0 mg/dL** | **≥ 5.5 – 8.5 mg/dL** | Unconjugated bilirubin indicates acute intravascular hemolysis. |
            | | **Serum Creatinine** | **0.50 – 0.65 mg/dL** | **0.65 – 0.80 mg/dL** | **≥ 1.05 – 1.40 mg/dL** | Creatinine rises from renal medullary ischemia / dehydration. |
            | **Clinical Action** | **Triage Protocol** | **Outpatient Home Protocol** (oral hydration, oral analgesia) | **ED Observation Unit** (60-min vitals, oral opioids, IV fluid) | **IMMEDIATE RESUSCITATION** (IV opioids in 30m, IV fluids, continuous SpO2) | Standard decision support integrated into SickleSense AI predictor. |
            """)

    # -------------------------------------------------------------
    # TAB 3: MODEL COMPARISON
    # -------------------------------------------------------------
    with tab_compare:
        st.subheader("📊 Model Benchmarks & Empirical Comparison")
        st.write(
            "Evaluation carried out using **Repeated Stratified 5-Fold Cross-Validation (25 rounds)** "
            "and **Leave-One-Out Cross-Validation (LOOCV)** on the real patient cohort ($N=23$). "
            "All metrics were evaluated strictly on **real held-out patients**."
        )

        st.markdown("#### 1. Real Data Baseline vs. Best Augmented Model")
        if df_rva is not None:
            st.dataframe(
                df_rva.style.format({
                    'Real_Only_Macro_F1': '{:.4f}',
                    'Real_Only_Std': '{:.4f}',
                    'Best_Augmented_Macro_F1': '{:.4f}',
                    'Best_Augmented_Std': '{:.4f}',
                    'Difference': '{:+.4f}'
                }).highlight_max(subset=['Best_Augmented_Macro_F1'], color='#dcfce7'),
                use_container_width=True
            )

        st.write("")
        st.markdown("#### 2. Real-Only Baseline Models (N=23)")
        if df_real is not None:
            st.dataframe(
                df_real.style.format({
                    'CV_Macro_F1_Mean': '{:.4f}',
                    'CV_Macro_F1_Std': '{:.4f}',
                    'CV_Bal_Acc_Mean': '{:.4f}',
                    'CV_Accuracy_Mean': '{:.4f}',
                    'LOOCV_Macro_F1': '{:.4f}',
                    'LOOCV_Accuracy': '{:.4f}'
                }).highlight_max(subset=['CV_Macro_F1_Mean', 'CV_Bal_Acc_Mean'], color='#dcfce7'),
                use_container_width=True
            )

        st.write("")
        st.markdown("#### 3. Visual Performance Comparison")
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            rva_chart = get_result_image('real_vs_augmented_f1.png')
            if rva_chart:
                st.image(rva_chart, caption="Real-Only vs. Best Augmented Macro F1 per Model", use_container_width=True)

        with col_c2:
            eff_chart = get_result_image('augmentation_effect.png')
            if eff_chart:
                st.image(eff_chart, caption="Effect of Synthetic Size on Real Patient Generalization", use_container_width=True)

        st.write("")
        col_c3, col_c4 = st.columns(2)

        with col_c3:
            f1_chart = get_result_image('macro_f1_comparison.png')
            if f1_chart:
                st.image(f1_chart, caption="Baseline Real-Data Macro F1 Comparison across 7 Algorithms", use_container_width=True)

        with col_c4:
            fi_chart = get_result_image('feature_importance.png')
            if fi_chart:
                st.image(fi_chart, caption="Physiological Feature Importance (RF Gini vs. Permutation)", use_container_width=True)

        st.write("")
        col_c5, col_c6 = st.columns(2)
        with col_c5:
            cm_priority = get_result_image('confusion_matrix_priority_augmented.png')
            if cm_priority:
                st.image(cm_priority, caption="Augmented Stacking Ensemble Confusion Matrix (5-Fold CV: 96.4% Accuracy)", use_container_width=True)

        with col_c6:
            cm_chart = get_result_image('confusion_matrix_stacking_ensemble.png')
            if cm_chart:
                st.image(cm_chart, caption="Baseline Stacking Ensemble Confusion Matrix (Real Patients: 27.0% Accuracy)", use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: ABOUT
    # -------------------------------------------------------------
    with tab_about:
        st.subheader("ℹ️ About the Study & Methodological Framework")

        with st.expander("📌 Project Overview & Dataset Scope", expanded=True):
            st.markdown("""
            - **Project Title**: Ensemble Machine Learning Approach for Predicting Pain Severity in Sickle Cell Disease
            - **Dataset Size**: Pilot cohort of 40 hospital patient records, yielding 23 fully labelled records for supervised training and 20 audit review cases.
            - **Target Definition**: Standardized into 3 clinical tiers:
              - **Mild**: Scores 0–3 or notes indicating mild/tolerable discomfort.
              - **Moderate**: Scores 4–6 or distressing pain.
              - **Severe**: Scores 7–10 or severe, incapacitating pain crisis.
            """)

        with st.expander("🛡️ Data Augmentation & Zero-Leakage Protocol", expanded=True):
            st.markdown(r"""
            - **Zero Data Leakage Protocol**:
              - Synthetic rows were generated **strictly inside each training fold** from training patients only (~18 patients).
              - Test folds contained **real, held-out patients only**.
              - No synthetic row was ever evaluated as a test instance or reported as a validation score.
            - **Augmentation Findings**:
              - SMOTE interpolation improved KNN Macro F1 from $0.1564 \rightarrow 0.2732$ by regularizing sparse local neighborhoods.
              - However, the observed gain ($+0.0559$) is smaller than the cross-validation standard deviation ($\pm 0.2144$), proving that generating synthetic rows from a tiny 18-patient training set does not introduce new biological diversity.
            """)

        with st.expander("⚠️ Clinical Study Limitations", expanded=True):
            st.markdown("""
            1. **Extremely Small Real Sample ($N=23$)**: Wide confidence intervals reflect real-world pilot sample constraints.
            2. **Age Heterogeneity**: Pediatric patients (down to age 4) and adult patients are assessed using identical baseline physiological thresholds.
            3. **Secondary Lab Missingness**: Features such as Serum Creatinine (39.1%) and Total Bilirubin (26.1%) required median imputation.
            4. **Subjective Ground Truth**: Target severity labels are derived from self-reported pain scores (0–10) and clinician notes.
            """)

        st.warning(
            "⚠️ **RESEARCH PROTOTYPE NOTICE**: This application is an investigational machine learning tool for research and educational purposes. "
            "It is **not** approved as a certified medical device and must **never** be used as a substitute for professional clinical judgment or emergency triage."
        )


if __name__ == '__main__':
    main()
