# train.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os

# Paths (relative to project root)
DATA_PATH = "data/Synthetic_SOAP_Dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "soap.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# 1. Load data
df = pd.read_csv(DATA_PATH)

# Safety check
df = df.dropna()
df = df.drop_duplicates()

X = df["sentence"]
y = df["label"]

# 2. Train/Test split (so we can measure performance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TF-IDF with n-grams
vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Class balancing
classes = df["label"].unique()
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=df["label"]
)
weights_dict = {cls: w for cls, w in zip(classes, class_weights)}

# 5. Train classifier
clf = LogisticRegression(
    class_weight=weights_dict,
    max_iter=500,
    solver="lbfgs"
)
clf.fit(X_train_vec, y_train)

# 6. Evaluate
y_pred = clf.predict(X_test_vec)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, digits=3))
print("\n🔍 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=classes))

# 7. Save models
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

with open(MODEL_PATH, "wb") as f:
    pickle.dump((vectorizer, clf), f)

print(f"\n✅ Model trained and saved to {MODEL_PATH}")
