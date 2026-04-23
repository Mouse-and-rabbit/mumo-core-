import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import joblib
import requests
from Bio.SeqUtils import ProtParam
from Bio.PDB import PDBParser, PPBuilder

# --- 1. SYSTEM INITIALIZATION (The Database) ---
MEMORY_FILE = 'protein_memory.json'
BRAIN_FILE = 'living_brain.pkl'
all_labels = np.array([0, 1])

# Load Brain & Cache (The persistent storage)
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

# --- FIX: consistent feature order ---
FEATURE_KEYS = [
    "molecular_weight",
    "aromaticity",
    "instability_index",
    "isoelectric_point",
    "gravy_score"
]

# --- 2. CORE BIOLOGICAL LOGIC ---

def extract_ai_features(sequence):
    """Converts protein letters into numerical bio-signatures."""
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

def predict_active_site(sequence):
    """Identifies potential catalytic residues."""
    critical_residues = ['H', 'C', 'D', 'E', 'S']
    found = [{"Residue": aa, "Position": i+1} for i, aa in enumerate(sequence) if aa in critical_residues]
    return pd.DataFrame(found)

def analyze_mutation(original_sequence, position, new_aa):
    """Predicts stability impact of a specific mutation."""
    mutant_list = list(original_sequence)

    # --- FIX: position safety ---
    if position < 1 or position > len(original_sequence):
        return 0, original_sequence

    mutant_list[position-1] = new_aa
    mutant_seq = "".join(mutant_list)

    orig_feats = extract_ai_features(original_sequence)
    mutant_feats = extract_ai_features(mutant_seq)

    if orig_feats is None or mutant_feats is None:
        return 0, mutant_seq

    orig_X = np.array([[orig_feats[k] for k in FEATURE_KEYS]])
    mutant_X = np.array([[mutant_feats[k] for k in FEATURE_KEYS]])

    # --- FIX: model not trained check ---
    if not hasattr(living_brain, "classes_"):
        return 0, mutant_seq

    orig_prob = living_brain.predict_proba(orig_X)[0][1]
    mutant_prob = living_brain.predict_proba(mutant_X)[0][1]

    return mutant_prob - orig_prob, mutant_seq

# --- 3. WEB INTERFACE ---
st.set_page_config(page_title="BioMumo (MOMU CORE)", layout="wide")
st.title("🧬 BioMumo: Adaptive Molecular Pipeline")

# SIDEBAR: Structural Input & Training
st.sidebar.header("Step 1: Structural Input")
input_method = st.sidebar.radio("Input Method", ["PDB ID", "Upload .PDB File", "Manual Sequence"])

final_seq = ""

if input_method == "PDB ID":
    pdb_id = st.sidebar.text_input("Enter 4-Letter PDB ID (e.g., 1A2B)")
    if pdb_id:
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        response = requests.get(url)

        # --- FIX: request failure handling ---
        if response.status_code == 200:
            with open("temp.pdb", "w") as f:
                f.write(response.text)

            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", "temp.pdb")
            ppb = PPBuilder()

            for pp in ppb.build_peptides(structure):
                final_seq += str(pp.get_sequence())
        else:
            st.error("❌ Failed to fetch PDB file.")

elif input_method == "Upload .PDB File":
    uploaded_file = st.sidebar.file_uploader("Choose a PDB file", type="pdb")
    if uploaded_file:
        with open("temp.pdb", "wb") as f:
            f.write(uploaded_file.getbuffer())

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", "temp.pdb")
        ppb = PPBuilder()

        for pp in ppb.build_peptides(structure):
            final_seq += str(pp.get_sequence())

else:
    final_seq = st.sidebar.text_area("Paste Sequence Here").strip().upper()

st.sidebar.divider()
st.sidebar.header("Step 2: Training Data")
actual_label = st.sidebar.selectbox("Known Stability?", [None, 0, 1], help="0=Unstable, 1=Stable")

# --- 4. MAIN DASHBOARD ---
if final_seq:
    st.subheader("Target Sequence")
    st.code(final_seq[:120] + "...")

    tab1, tab2, tab3 = st.tabs(["Stability Prediction", "Active Site Map", "Mutation Lab"])

    with tab1:
        if st.button("Run BioMumo Prediction"):
            if final_seq in fast_cache:
                st.success(f"📍 [MEMORY MATCH] Result: {fast_cache[final_seq]['verdict']}")
            else:
                feats = extract_ai_features(final_seq)

                # --- FIX: feature extraction check ---
                if feats is None:
                    st.error("❌ Feature extraction failed.")
                else:
                    X = np.array([[feats[k] for k in FEATURE_KEYS]])

                    # --- FIX: model not trained ---
                    if not hasattr(living_brain, "classes_"):
                        st.warning("⚠️ Model not trained yet. Provide a labeled example.")
                    else:
                        pred = living_brain.predict(X)[0]
                        verdict = "STABLE/ACTIVE" if pred == 1 else "UNSTABLE/INACTIVE"
                        st.info(f"🔍 [PREDICTION] AI Estimates: {verdict}")

                        # Live Learning
                        if actual_label is not None:
                            living_brain.partial_fit(X, np.array([actual_label]), classes=all_labels)
                            fast_cache[final_seq] = {"verdict": verdict}

                            joblib.dump(living_brain, BRAIN_FILE)
                            with open(MEMORY_FILE, 'w') as f:
                                json.dump(fast_cache, f)

                            st.write("🧠 Brain Updated & Saved Locally.")

    with tab2:
        st.subheader("Catalytic Site Mapping")
        st.dataframe(predict_active_site(final_seq))

    with tab3:
        st.subheader("In-Silico Mutagenesis")
        pos = st.number_input("Position to Mutate", min_value=1, max_value=len(final_seq), value=1)
        new_aa = st.selectbox("New Amino Acid", list("ACDEFGHIKLMNPQRSTVWY"))

        if st.button("Predict Mutation Impact"):
            diff, m_seq = analyze_mutation(final_seq, pos, new_aa)
            st.metric("Stability Change Score", f"{diff*100:.2f}%", delta=f"{diff:.4f}")