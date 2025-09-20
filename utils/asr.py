# utils/asr.py
import os, tempfile
from faster_whisper import WhisperModel

def get_asr(model_size="small"):
    return WhisperModel(model_size, device="cpu", compute_type="int8")

def transcribe_audio(file_bytes: bytes, model_size="small") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        model = get_asr(model_size)
        segments, _ = model.transcribe(tmp_path, vad_filter=True)
        return " ".join(s.text.strip() for s in segments)
    finally:
        try: os.remove(tmp_path)
        except: pass
