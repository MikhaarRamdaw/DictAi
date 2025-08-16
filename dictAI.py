# app.py
# Run: streamlit run app.py

import os, re, joblib, streamlit as st
from pathlib import Path
from typing import List, Dict
from st_audiorec import st_audiorec
from faster_whisper import WhisperModel
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
import tempfile

# -----------------------------
# Paths & constants
# -----------------------------
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "soap.pkl"
LE_PATH = MODEL_DIR / "label_encoder.pkl"

LABELS = ["subjective", "objective", "assessment", "plan"]

# -----------------------------
# Utilities
# -----------------------------
def split_sentences(text: str) -> List[str]:
    t = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+|;\s+|\n+", t)
    return [p.strip() for p in parts if p.strip()]

@st.cache_resource
def load_or_init_model():
    if MODEL_PATH.exists() and LE_PATH.exists():
        pipe = joblib.load(MODEL_PATH)
        le = joblib.load(LE_PATH)
        return pipe, le
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1,2),
            min_df=1,
            max_features=50000
        )),
        ("clf", SGDClassifier(loss="log_loss", max_iter=5, tol=None))
    ])
    le = LabelEncoder().fit(LABELS)
    return pipe, le

@st.cache_resource
def get_asr(model_size: str = "small"):
    # Use device="cuda" if you have a GPU
    return WhisperModel(model_size, device="cpu", compute_type="int8")

def transcribe_audio(file_bytes: bytes, model_size="small") -> str:
    # Write to a temporary file and let faster-whisper handle it
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

def cold_start_train(pipe, le, sentences, labels):
    y = le.transform(labels)
    # Register classes once for partial_fit
    pipe.named_steps["clf"].partial_fit(
        np.zeros((1,1)), np.array([0]), classes=np.arange(len(LABELS))
    )
    X = pipe.named_steps["tfidf"].fit_transform(sentences)
    pipe.named_steps["clf"].partial_fit(X, y)
    return pipe

def predict_with_conf(pipe, le, sentences):
    X = pipe.named_steps["tfidf"].transform(sentences)
    proba = pipe.named_steps["clf"].predict_proba(X)
    y_pred = np.argmax(proba, axis=1)
    labels = le.inverse_transform(y_pred)
    conf = proba.max(axis=1)
    return labels, conf

def online_update(pipe, le, sentences, labels):
    X = pipe.named_steps["tfidf"].transform(sentences)
    y = le.transform(labels)
    pipe.named_steps["clf"].partial_fit(X, y)
    return pipe

def render_markdown(soap: Dict[str, List[str]]) -> str:
    def sec(title, lines):
        if not lines: return f"**{title}**\n\n"
        body = "\n".join(lines) if title != "Plan" else "\n".join(f"- {ln}" for ln in lines)
        return f"**{title}**\n{body}\n\n"
    return (
        "# SOAP Note\n\n" +
        sec("Subjective", soap["subjective"]) +
        sec("Objective",  soap["objective"]) +
        sec("Assessment", soap["assessment"]) +
        sec("Plan",       soap["plan"])
    )

# -----------------------------
# UI
# -----------------------------
st.title("Dictation → SOAP (Speak or Upload)")

with st.sidebar:
    st.markdown("### Workflow")
    st.markdown("1) **Record** or **Upload** audio\n2) Transcribe\n3) Auto-label sentences\n4) Fix mistakes\n5) Save & Learn")
    st.markdown("---")
    st.markdown("**Tip:** First time, label ~12–20 sentences to train the model.")

model_size = st.selectbox("ASR model size", ["tiny", "base", "small", "medium"], index=2)

# ---- Microphone recording via st_audiorec
st.subheader("🎤 Record audio")
rec_bytes = st_audiorec()   # Returns WAV bytes or None
col_rec1, col_rec2 = st.columns([1,4])
with col_rec2:
    if rec_bytes:
        st.audio(rec_bytes, format="audio/wav")
with col_rec1:
    rec_btn = st.button("Transcribe recording")

# ---- File upload alternative
st.subheader("📁 Or upload audio file")
audio = st.file_uploader("Upload audio (wav/mp3/m4a)", type=["wav","mp3","m4a"])

# ---- Manual transcript (optional)
st.subheader("📝 Or paste transcript")
raw_text = st.text_area("Transcript (optional if recording/uploading)", height=150, placeholder="Paste text here...")

pipe, le = load_or_init_model()

# ---- Actions: Transcribe
if rec_btn and rec_bytes:
    with st.spinner("Transcribing your recording..."):
        raw_text = transcribe_audio(rec_bytes, model_size=model_size)
        st.success("Recording transcribed ✓")

if st.button("Transcribe uploaded file") and audio is not None:
    with st.spinner("Transcribing your file..."):
        raw_text = transcribe_audio(audio.read(), model_size=model_size)
        st.success("File transcribed ✓")

# Show transcript
if raw_text:
    st.markdown("#### Transcript")
    st.write(raw_text)

# -----------------------------
# Labeling & Learning
# -----------------------------
if raw_text:
    sents = split_sentences(raw_text)

    st.subheader("🔖 Sentence labelling")
    if not MODEL_PATH.exists():
        st.info("First run: please label ~12–20 sentences to teach the model.")
        labels = []
        max_to_label = min(20, len(sents))
        for i, s in enumerate(sents[:max_to_label]):
            lab = st.selectbox(f"Sentence {i+1}: {s}", LABELS, key=f"cold{i}")
            labels.append(lab)
        if st.button("Train initial model"):
            pipe = cold_start_train(pipe, le, sents[:len(labels)], labels)
            joblib.dump(pipe, MODEL_PATH); joblib.dump(le, LE_PATH)
            st.success("Model trained ✓ You can now auto-label the rest.")
    else:
        pred_labels, conf = predict_with_conf(pipe, le, sents)
        edited = []
        for i, (s, lab, c) in enumerate(zip(sents, pred_labels, conf)):
            col1, col2, col3 = st.columns([6,2,1])
            with col1: st.write(s)
            with col2:
                sel = st.selectbox("Label", LABELS, index=LABELS.index(lab), key=f"lab{i}")
            with col3:
                st.caption(f"conf {c:.2f}")
            edited.append((s, sel))

        if st.button("Apply & Learn"):
            grouped = {"subjective": [], "objective": [], "assessment": [], "plan": []}
            for s, lab in edited:
                grouped[lab].append(s)
            md = render_markdown(grouped)
            st.download_button("Download SOAP.md", md, file_name="soap_note.md")
            pipe = online_update(pipe, le, [s for s,_ in edited], [lab for _,lab in edited])
            joblib.dump(pipe, MODEL_PATH); joblib.dump(le, LE_PATH)
            st.success("Learned from your edits ✓ Future notes will be better.")
