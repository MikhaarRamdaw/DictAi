# app.py
# Run with: streamlit run app.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import os
import re
import joblib
import streamlit as st
import tempfile
from pathlib import Path
from typing import List, Dict
from streamlit_mic_recorder import mic_recorder
from faster_whisper import WhisperModel
from utils.render import render_markdown, render_pdf

# -----------------------------
# Paths & constants
# -----------------------------
MODEL_PATH = Path("models/soap.pkl")

LABELS = ["subjective", "objective", "assessment", "plan"]

# -----------------------------
# Utilities
# -----------------------------
def split_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    t = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+|;\s+|\n+", t)
    return [p.strip() for p in parts if p.strip()]

@st.cache_resource
def load_model():
    """Load trained TF-IDF + classifier pipeline."""
    if not MODEL_PATH.exists():
        st.error("⚠️ No trained model found. Please run train.py first.")
        st.stop()
    vectorizer, clf = joblib.load(MODEL_PATH)
    return vectorizer, clf

@st.cache_resource
def get_asr(model_size: str = "small"):
    """Load Whisper ASR model."""
    return WhisperModel(model_size, device="cpu", compute_type="int8")

def transcribe_audio(file_bytes: bytes, model_size="small") -> str:
    """Transcribe audio file using faster-whisper."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        model = get_asr(model_size)
        segments, _ = model.transcribe(tmp_path, vad_filter=True)
        return " ".join(s.text.strip() for s in segments)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def predict_sentences(vectorizer, clf, sentences: List[str]):
    """Classify sentences into SOAP labels."""
    X = vectorizer.transform(sentences)
    preds = clf.predict(X)
    return list(zip(sentences, preds))

def group_to_soap(preds: List[tuple]) -> Dict[str, List[str]]:
    """Group predictions into SOAP sections."""
    grouped = {lab: [] for lab in LABELS}
    for s, lab in preds:
        grouped[lab].append(s)
    return grouped

# -----------------------------
# UI
# -----------------------------
st.title("🩺 Dictation → SOAP Notes (Auto)")

with st.sidebar:
    st.markdown("### Workflow")
    st.markdown("1. 🎤 Record or 📁 Upload audio, or 📝 Paste text")
    st.markdown("2. 🔊 Transcribe")
    st.markdown("3. 🤖 Auto-classify into SOAP")
    st.markdown("4. 📄 Export as PDF/Markdown")

# Select ASR model size
model_size = st.selectbox("ASR model size", ["tiny", "base", "small", "medium"], index=2)

# ---- Microphone recording
st.subheader("🎤 Record audio")
rec_bytes = mic_recorder(start_prompt="Start recording", stop_prompt="Stop recording")
if rec_bytes:
    st.audio(rec_bytes["bytes"], format="audio/wav")

if st.button("Transcribe recording") and rec_bytes:
    with st.spinner("Transcribing your recording..."):
        raw_text = transcribe_audio(rec_bytes["bytes"], model_size=model_size)
        st.session_state["transcript"] = raw_text
        st.success("Recording transcribed ✓")

# ---- File upload
st.subheader("📁 Or upload audio file")
audio = st.file_uploader("Upload audio (wav/mp3/m4a)", type=["wav","mp3","m4a"])
if st.button("Transcribe uploaded file") and audio:
    with st.spinner("Transcribing your file..."):
        raw_text = transcribe_audio(audio.read(), model_size=model_size)
        st.session_state["transcript"] = raw_text
        st.success("File transcribed ✓")

# ---- Manual text
st.subheader("📝 Or paste transcript")
raw_text = st.text_area("Transcript", value=st.session_state.get("transcript", ""), height=150)

# -----------------------------
# Prediction & Output
# -----------------------------
if raw_text:
    st.markdown("#### Transcript")
    st.write(raw_text)

    vectorizer, clf = load_model()
    sentences = split_sentences(raw_text)
    preds = predict_sentences(vectorizer, clf, sentences)
    grouped = group_to_soap(preds)

    # Show structured SOAP note
    st.markdown("### 📝 Generated SOAP Note")
    md = render_markdown(grouped)
    st.markdown(md)

    # Export options
    st.download_button("⬇️ Download as Markdown", md, file_name="soap_note.md")

    pdf_bytes = render_pdf(grouped)
    st.download_button("⬇️ Download as PDF", pdf_bytes, file_name="soap_note.pdf", mime="application/pdf")
