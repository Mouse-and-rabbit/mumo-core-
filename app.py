import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import joblib
import requests
from Bio.SeqUtils import ProtParam
from Bio.PDB import PDBParser, PPBuilder

# --- 1. SYSTEM INITIALIZATION ---
MEMORY_FILE = 'protein_memory.json'
BRAIN_FILE = 'living_brain.pkl'
all_labels = np.array([0, 1])

if os.path.exists(BRAIN_FILE):
    living_brain = joblib.load(BRAIN_FILE)
else:
    from sklearn.linear_model import SGDClassifier
    living_brain = SGDClassifier(loss='log_loss', random_state=42)

if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, 'r') as f:
        fast_cache = json.load(f)
else:
    fast_cache = {}

FEATURE_KEYS = [
    "molecular_weight",
    "aromaticity",
    "instability_index",
    "isoelectric_point",
    "gravy_score"
]

# --- 2. CORE LOGIC ---
def extract_ai_features(sequence):
    try:
        analysis = ProtParam.ProteinAnalysis(sequence)
        return {
            "molecular_weight": analysis.molecular_weight(),
            "aromaticity": analysis.aromaticity(),
            "instability_index": analysis.instability_index(),
            "isoelectric_point": analysis.isoelectric_point(),
            "gravy_score": analysis.gravy(),
        }
    except:
        return None

# --- PIPELINE FUNCTION (WITH CHARTS) ---
def run_pipeline(sequence):
    feats = extract_ai_features(sequence)

    if feats is None:
        st.error("❌ Feature extraction failed.")
        return

    X = np.array([[feats[k] for k in FEATURE_KEYS]])

    if not hasattr(living_brain, "classes_"):
        st.warning("⚠️ Model not trained yet.")
        return

    pred = living_brain.predict(X)[0]
    prob = living_brain.predict_proba(X)[0][1]

    verdict = "STABLE/ACTIVE" if pred == 1 else "UNSTABLE/INACTIVE"

    # --- Charts ---
    st.subheader("📊 Stability Prediction")
    chart_data = pd.DataFrame({
        "Category": ["Unstable", "Stable"],
        "Probability": [1 - prob, prob]
    })
    st.bar_chart(chart_data.set_index("Category"))

    st.subheader("📉 Biophysical Features")
    feature_df = pd.DataFrame({
        "Feature": FEATURE_KEYS,
        "Value": [feats[k] for k in FEATURE_KEYS]
    })
    st.bar_chart(feature_df.set_index("Feature"))

    st.info(f"Final Prediction: {verdict}")

# --- 3. UI SETUP ---
st.set_page_config(page_title="BioMumo (MOMU CORE)", layout="wide")
st.title("🧬 BioMumo: Adaptive Molecular Pipeline")

# --- SIDEBAR INPUT ---
st.sidebar.header("Step 1: Input")
input_method = st.sidebar.radio("Input Method", ["PDB ID", "Upload .PDB File", "Manual Sequence"])

final_seq = ""

if input_method == "PDB ID":
    pdb_id = st.sidebar.text_input("Enter PDB ID")
    if pdb_id:
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        response = requests.get(url)

        if response.status_code == 200:
            with open("temp.pdb", "w") as f:
                f.write(response.text)

            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", "temp.pdb")
            ppb = PPBuilder()

            for pp in ppb.build_peptides(structure):
                final_seq += str(pp.get_sequence())
        else:
            st.error("Failed to fetch PDB")

elif input_method == "Upload .PDB File":
    file = st.sidebar.file_uploader("Upload PDB", type="pdb")
    if file:
        with open("temp.pdb", "wb") as f:
            f.write(file.getbuffer())

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", "temp.pdb")
        ppb = PPBuilder()

        for pp in ppb.build_peptides(structure):
            final_seq += str(pp.get_sequence())

else:
    final_seq = st.sidebar.text_area("Paste Sequence").strip().upper()

# --- MAIN UI ---
if final_seq:
    st.subheader("Target Sequence")
    st.code(final_seq[:120] + "...")

    st.markdown("""
    <style>
    .card {
        border: 2px solid #2d3436;
        border-radius: 15px;
        padding: 20px;
        background: white;
        box-shadow: 4px 4px 0px #2d3436;
        text-align: center;
        height: 320px;
    }
    .card ul {
        text-align: left;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    # --- CARD 1 ---
    with c1:
        st.markdown("""<div class="card">
        <h3>🔬 Structural Analysis</h3>
        <ul>
            <li>PDB loading</li>
            <li>Sequence extraction</li>
            <li>Backbone parsing</li>
            <li>3D coordinates</li>
            <li>Preprocessing</li>
        </ul></div>""", unsafe_allow_html=True)

        if st.button("Run", key="b1"):
            run_pipeline(final_seq)

    # --- CARD 2 ---
    with c2:
        st.markdown("""<div class="card">
        <h3>🧮 Feature Engineering</h3>
        <ul>
            <li>Molecular weight</li>
            <li>Instability index</li>
            <li>pI</li>
            <li>GRAVY</li>
            <li>Feature extraction</li>
        </ul></div>""", unsafe_allow_html=True)

        if st.button("Run", key="b2"):
            run_pipeline(final_seq)

    # --- CARD 3 ---
    with c3:
        st.markdown("""<div class="card">
        <h3>🧠 AI Prediction</h3>
        <ul>
            <li>Stability prediction</li>
            <li>Learning model</li>
            <li>Classification</li>
            <li>Probability scoring</li>
            <li>Output analysis</li>
        </ul></div>""", unsafe_allow_html=True)

        if st.button("Run", key="b3"):
            run_pipeline(final_seq)
