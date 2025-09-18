

---

# 🩺 DrAI – Development Roadmap

DrAI is an offline-first SOAP notes dictation assistant. Below is the **outline of tasks** our team needs to code and integrate, step by step.

---

## 1. **Dataset Handling**

* **File:** `data/synthetic_soap.csv`
* **Task:**

  * Load dataset into Pandas (`pd.read_csv`).
  * Ensure it has two columns: `sentence`, `label`.
  * Add function to preprocess text (strip whitespace, lowercasing if needed).
* **Output:**

  * Cleaned sentences + labels ready for training.

---

## 2. **Model Training**

* **File:** `train.py`
* **Task:**

  * Import dataset.
  * Encode labels using `LabelEncoder`.
  * Train classifier:

    ```python
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=50000)),
        ("clf", SGDClassifier(loss="log_loss", max_iter=1000))
    ])
    pipe.fit(sentences, labels)
    ```
  * Save artifacts with `joblib.dump`:

    * `models/soap.pkl`
    * `models/label_encoder.pkl`
* **Output:**

  * Pretrained classifier + encoder, available for app.

---

## 3. **Model Loading**

* **File:** `app.py`
* **Task:**

  * Write function `load_model()` to load `soap.pkl` and `label_encoder.pkl`.
  * If not found, display error message to team: “Train the model first.”
* **Output:**

  * App can use pretrained classifier immediately.

---

## 4. **Speech-to-Text (ASR)**

* **File:** `app.py`
* **Task:**

  * Use `faster-whisper` to transcribe uploaded/recorded audio.
  * Write helper function `transcribe_audio(file_bytes, model_size="small")`.
* **Output:**

  * Transcript string from raw audio.

---

## 5. **Sentence Splitting**

* **File:** `utils/text_utils.py` (new helper file)
* **Task:**

  * Write function `split_sentences(text)` using regex.
* **Output:**

  * Clean list of sentences for classification.

---

## 6. **Classification & Confidence**

* **File:** `app.py`
* **Task:**

  * Add function `predict_with_conf(pipe, le, sentences)`.
  * Should return predicted labels + confidence scores.
* **Output:**

  * Each sentence labelled as Subjective / Objective / Assessment / Plan.

---

## 7. **SOAP Rendering**

* **File:** `utils/render.py`
* **Task:**

  * Group sentences by label.
  * Format into Markdown:

    ```md
    # SOAP Note

    **Subjective**
    <sentences>

    **Objective**
    <sentences>

    ...
    ```
* **Output:**

  * Ready-to-download formatted note.

---

## 8. **Streamlit Frontend**

* **File:** `app.py`
* **Task:**

  * Sidebar: instructions + ASR model size selector.
  * Main page:

    * Record audio (`st_audiorec`).
    * Upload audio file.
    * Paste transcript (manual).
  * After transcription:

    * Display transcript.
    * Auto-classify into SOAP.
    * Show predictions + confidence.
  * Add download buttons for Markdown (later PDF/JSON).
* **Output:**

  * End-to-end doctor workflow: Speak → Transcribe → Classify → Export.

---

## 9. **Export Options**

* **File:** `utils/export.py`
* **Task:**

  * `export_md()` → save Markdown.
  * `export_pdf()` → generate PDF with `reportlab`.
  * (Optional) `export_json()` for EHR integration.
* **Output:**

  * Doctor can save SOAP notes in multiple formats.

---

## 10. **Deployment**

* **Task:**

  * Add `requirements.txt`.
  * Build local executable with PyInstaller:

    ```bash
    pyinstaller --onefile app.py --add-data "models:models"
    ```
  * Test offline run.
* **Output:**

  * Share `.exe` (Windows) / `.app` (Mac) with doctors.

---

## 📝 Development Checklist

* [ ] Dataset preprocessing (`data/synthetic_soap.csv`).
* [ ] Training script (`train.py`).
* [ ] Model loading in `app.py`.
* [ ] Whisper transcription integration.
* [ ] Sentence splitting helper.
* [ ] Classification + confidence function.
* [ ] SOAP Markdown rendering.
* [ ] Streamlit UI (record/upload/paste → classify → export).
* [ ] Export functions (MD, PDF, JSON).
* [ ] Deployment (packaging + installer).

---


Do you want me to also **write the `train.py` skeleton** with TODOs for the team (so each person can just fill in their part), or should I keep it as a high-level outline?
