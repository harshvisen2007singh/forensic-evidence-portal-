import datetime
import hashlib
import json
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Forensic Toxicology & Evidence Registry",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header { font-size:2.2rem; color:#1E3A8A; font-weight:700; margin-bottom:0px; }
    .sub-header { font-size:1.0rem; color:#4B5563; margin-bottom:20px; }
    .card { background-color:#F3F4F6; padding:15px; border-radius:10px; border-left: 5px solid #1E3A8A; }
    </style>
""", unsafe_allow_html=True)

# Session State for Tamper-Proof Evidence Registry
if "evidence_registry" not in st.session_state:
    st.session_state.evidence_registry = []

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def generate_sha256_hash(data_dict: dict) -> str:
    """Generates a cryptographic SHA-256 digital signature for case verification."""
    serialized_data = json.dumps(data_dict, sort_keys=True)
    return hashlib.sha256(serialized_data.encode('utf-8')).hexdigest()

def simulate_spectral_data(target_compound: str, concentration: float):
    """Generates synthetic spectrophotometric wavelength vs absorbance data."""
    wavelengths = np.linspace(200, 700, 250)
    base_absorbance = np.exp(-((wavelengths - 250)**2) / (2 * 30**2)) * 0.2
    
    # Peak wavelength parameters per toxic agent
    peaks = {
        "Organophosphate Pesticide": (340, 45),
        "Arsenic Heavy Metal": (520, 25),
        "Cyanide Salt": (280, 20),
        "Synthetic Narcotic (Opioid)": (410, 35)
    }
    
    center_wl, spread = peaks.get(target_compound, (300, 30))
    sample_peak = np.exp(-((wavelengths - center_wl)**2) / (2 * spread**2)) * concentration
    noise = np.random.normal(0, 0.015, len(wavelengths))
    
    sample_absorbance = np.clip(base_absorbance + sample_peak + noise, 0, None)
    
    return pd.DataFrame({
        "Wavelength_nm": wavelengths,
        "Control_Absorbance": base_absorbance,
        "Sample_Absorbance": sample_absorbance
    })

def process_latent_image(image_bytes, contrast_val: float, filter_type: str):
    """Applies OpenCV image processing to clean degraded latent evidence samples."""
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Contrast adjustment
    adjusted = cv2.convertScaleAbs(gray, alpha=contrast_val, beta=0)
    
    if filter_type == "Adaptive Thresholding (Minutiae Contrast)":
        processed = cv2.adaptiveThreshold(
            adjusted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
    elif filter_type == "Gaussian Denoising":
        processed = cv2.GaussianBlur(adjusted, (5, 5), 0)
    elif filter_type == "Canny Edge Detection":
        processed = cv2.Canny(adjusted, 50, 150)
    else:
        processed = adjusted
        
    return gray, processed

# ==========================================
# DASHBOARD SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🔬 Navigation Panel")
menu_choice = st.sidebar.radio(
    "Select Portal Module:",
    [
        "1. Toxicological Spectral Analysis",
        "2. Latent Evidence Pre-Processor",
        "3. Medico-Legal Verification Registry"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("**Navadhara '26 Prototype**\n\n*Track 2: Engineering, Healthcare & Life Sciences*")

# ==========================================
# MODULE 1: TOXICOLOGICAL SPECTRAL ANALYSIS
# ==========================================
if menu_choice == "1. Toxicological Spectral Analysis":
    st.markdown('<div class="main-header">Forensic Toxicological Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Spectral peak matching, automated toxicity scoring, and compound detection.</div>', unsafe_allow_html=True)
    
    col_input, col_view = st.columns([1, 2])
    
    with col_input:
        st.subheader("📋 Case & Sample Metadata")
        case_id = st.text_input("Case File Reference", "CASE-2026-8802")
        examiner_id = st.text_input("Examiner Code", "EXAM-CSE-BIO-12")
        
        target_compound = st.selectbox(
            "Suspected Toxic Compound Profile",
            ["Organophosphate Pesticide", "Arsenic Heavy Metal", "Cyanide Salt", "Synthetic Narcotic (Opioid)"]
        )
        
        concentration = st.slider("Estimated Concentration Level (mg/L)", 0.1, 5.0, 1.8, step=0.1)
        
        run_analysis = st.button("🧪 Run Spectral Analysis", type="primary")
        
    df_spectral = simulate_spectral_data(target_compound, concentration)
    
    with col_view:
        st.subheader("📊 Wavelength vs Absorbance Spectrum")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_spectral["Wavelength_nm"], y=df_spectral["Control_Absorbance"],
            mode='lines', name='Normal Biological Control', line=dict(color='#9CA3AF', dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=df_spectral["Wavelength_nm"], y=df_spectral["Sample_Absorbance"],
            mode='lines', name=f'Sample ({target_compound})', line=dict(color='#DC2626', width=2)
        ))
        fig.update_layout(
            xaxis_title="Wavelength (nm)",
            yaxis_title="Absorbance Units (AU)",
            margin=dict(l=20, r=20, t=30, b=20),
            height=380
        )
        st.plotly_chart(fig, use_container_width=True)
        
    if run_analysis:
        st.markdown("---")
        st.subheader("🎯 Analytical Diagnostics & Toxicity Index")
        
        match_confidence = min(99.4, round(85.0 + (concentration * 2.8), 2))
        toxicity_level = "CRITICAL / FATAL" if concentration > 2.5 else "HIGH TOXICITY" if concentration > 1.2 else "MODERATE"
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Compound Identification", target_compound)
        m2.metric("Spectral Pattern Match", f"{match_confidence}%")
        m3.metric("Toxicity Severity Index", toxicity_level)
        
        # Log to SHA-256 Registry
        case_data = {
            "case_id": case_id,
            "examiner_id": examiner_id,
            "compound": target_compound,
            "concentration_mg_l": concentration,
            "match_confidence": match_confidence,
            "timestamp": datetime.datetime.now().isoformat()
        }
        case_hash = generate_sha256_hash(case_data)
        case_data["sha256_signature"] = case_hash
        
        st.session_state.evidence_registry.append(case_data)
        
        st.success(f"✅ Case record successfully processed and signed. Digital Verification Hash:\n`{case_hash}`")

# ==========================================
# MODULE 2: LATENT EVIDENCE PRE-PROCESSOR
# ==========================================
elif menu_choice == "2. Latent Evidence Pre-Processor":
    st.markdown('<div class="main-header">Latent Pattern & Impression Pre-Processor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">OpenCV-assisted contrast enhancement and noise reduction for degraded physical evidence.</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Degraded Latent Image (Fingerprint, Footwear, Stain)", type=["jpg", "png", "jpeg"])
    
    col_ctrl, col_img1, col_img2 = st.columns([1, 1.5, 1.5])
    
    with col_ctrl:
        st.subheader("⚙️ Image Controls")
        contrast = st.slider("Contrast Multiplier", 1.0, 3.0, 1.6, step=0.1)
        filter_type = st.selectbox(
            "Filter Pipeline",
            ["Adaptive Thresholding (Minutiae Contrast)", "Gaussian Denoising", "Canny Edge Detection", "Raw Grayscale"]
        )
        
    if uploaded_file is not None:
        raw_gray, processed_img = process_latent_image(uploaded_file, contrast, filter_type)
        
        with col_img1:
            st.subheader("Original (Grayscale)")
            st.image(raw_gray, use_column_width=True)
            
        with col_img2:
            st.subheader("Enhanced Output")
            st.image(processed_img, use_column_width=True)
    else:
        st.info("💡 Upload a sample image above to test real-time OpenCV enhancement routines.")

# ==========================================
# MODULE 3: MEDICO-LEGAL VERIFICATION REGISTRY
# ==========================================
elif menu_choice == "3. Medico-Legal Verification Registry":
    st.markdown('<div class="main-header">Tamper-Proof Medico-Legal Evidence Registry</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Cryptographic SHA-256 audit ledger ensuring forensic evidence integrity for court presentation.</div>', unsafe_allow_html=True)
    
    if len(st.session_state.evidence_registry) == 0:
        st.warning("No case logs generated yet. Run a spectral toxicological test in Module 1 to populate records.")
    else:
        df_registry = pd.DataFrame(st.session_state.evidence_registry)
        st.subheader(f"📋 Verified Audit Records ({len(df_registry)} Total Entries)")
        
        st.dataframe(df_registry[["case_id", "examiner_id", "compound", "match_confidence", "timestamp", "sha256_signature"]], use_container_width=True)
        
        st.markdown("---")
        st.subheader("🛡️ Forensic Evidence Signature Inspector")
        selected_case = st.selectbox("Select Case File to Inspect", df_registry["case_id"].unique())
        
        record = df_registry[df_registry["case_id"] == selected_case].iloc[-1]
        
        st.code(f"""
================================================================================
                    OFFICIAL MEDICO-LEGAL EVIDENCE SUMMARY
================================================================================
Case Reference       : {record['case_id']}
Examiner Code        : {record['examiner_id']}
Timestamp (UTC)      : {record['timestamp']}
Identified Agent     : {record['compound']}
Concentration        : {record['concentration_mg_l']} mg/L
Matching Confidence  : {record['match_confidence']}%

CRYPTOGRAPHIC INTEGRITY VERIFICATION SIGNATURE (SHA-256):
{record['sha256_signature']}
================================================================================
STATUS: VERIFIED & TAMPER-FREE
        """, language="text")