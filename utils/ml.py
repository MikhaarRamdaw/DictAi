# utils/ml.py
import joblib
import numpy as np
from pathlib import Path

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "soap.pkl"
LE_PATH = MODEL_DIR / "label_encoder.pkl"

def load_model():
    pipe = joblib.load(MODEL_PATH)
    le = joblib.load(LE_PATH)
    return pipe, le

def predict_with_conf(pipe, le, sentences):
    X = pipe.named_steps["tfidf"].transform(sentences)
    proba = pipe.named_steps["clf"].predict_proba(X)
    y_pred = np.argmax(proba, axis=1)
    labels = le.inverse_transform(y_pred)
    conf = proba.max(axis=1)
    return labels, conf
