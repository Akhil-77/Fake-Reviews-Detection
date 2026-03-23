"""
app.py
======
Flask web application for fake-review detection.

Supports all 5 models:
  - Logistic Regression
  - Random Forest
  - LSTM
  - BiLSTM
  - BERT
  - RoBERTa

Run locally:
    python app.py

For free GitHub Pages / Render / Hugging Face Spaces deployment see README.md.
"""

import os, re, pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

# ── lazy-load heavy deps only when the relevant model is first requested ──────
_tfidf        = None
_tokenizer    = None
_models_cache = {}

ARTIFACT_DIR  = "artifacts"
MAX_SEQ_LEN   = 200          # must match data_prep.py
DEVICE        = "cpu"        # keep CPU for free-tier hosting

app = Flask(__name__)


# ── text cleaning (identical to data_prep.py) ─────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── artefact loaders ──────────────────────────────────────────────────────────
def get_tfidf():
    global _tfidf
    if _tfidf is None:
        with open(f"{ARTIFACT_DIR}/tfidf_vectorizer.pkl", "rb") as f:
            _tfidf = pickle.load(f)
    return _tfidf


def get_keras_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        with open(f"{ARTIFACT_DIR}/tokenizer.pkl", "rb") as f:
            _tokenizer = pickle.load(f)
    return _tokenizer


def get_model(model_key: str):
    if model_key in _models_cache:
        return _models_cache[model_key]

    if model_key == "logistic_regression":
        import pickle
        with open(f"{ARTIFACT_DIR}/logistic_regression.pkl", "rb") as f:
            m = pickle.load(f)

    elif model_key == "random_forest":
        import pickle
        with open(f"{ARTIFACT_DIR}/random_forest.pkl", "rb") as f:
            m = pickle.load(f)

    elif model_key == "lstm":
        import tensorflow as tf
        m = tf.keras.models.load_model(f"{ARTIFACT_DIR}/lstm_model")

    elif model_key == "bilstm":
        import tensorflow as tf
        m = tf.keras.models.load_model(f"{ARTIFACT_DIR}/bilstm_model")

    elif model_key in ("bert", "roberta"):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        dir_name = "bert_model" if model_key == "bert" else "roberta_model"
        tok = AutoTokenizer.from_pretrained(f"{ARTIFACT_DIR}/{dir_name}")
        mdl = AutoModelForSequenceClassification.from_pretrained(
            f"{ARTIFACT_DIR}/{dir_name}"
        ).to(DEVICE)
        mdl.eval()
        m = (tok, mdl)   # tuple for transformer models

    else:
        raise ValueError(f"Unknown model: {model_key}")

    _models_cache[model_key] = m
    return m


# ── prediction helpers ────────────────────────────────────────────────────────
def predict_classical(model_key: str, text: str) -> dict:
    vec   = get_tfidf().transform([text])
    model = get_model(model_key)
    prob  = model.predict_proba(vec)[0, 1]
    label = "real" if prob > 0.5 else "fake"
    return {"label": label, "confidence": round(float(prob if label == "real" else 1 - prob), 4)}


def predict_deep(model_key: str, text: str) -> dict:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    tok = get_keras_tokenizer()
    seq = pad_sequences(
        tok.texts_to_sequences([text]),
        maxlen=MAX_SEQ_LEN, padding="post", truncating="post"
    )
    model = get_model(model_key)
    prob  = float(model.predict(seq, verbose=0)[0, 0])
    label = "real" if prob > 0.5 else "fake"
    return {"label": label, "confidence": round(prob if label == "real" else 1 - prob, 4)}


def predict_transformer(model_key: str, text: str) -> dict:
    import torch
    tokenizer, model = get_model(model_key)
    enc = tokenizer(
        text,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE),
        ).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    idx   = int(np.argmax(probs))
    label = "real" if idx == 1 else "fake"
    return {"label": label, "confidence": round(float(probs[idx]), 4)}


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data       = request.get_json(force=True)
    raw_text   = data.get("text", "").strip()
    model_key  = data.get("model", "logistic_regression").strip().lower()

    if not raw_text:
        return jsonify({"error": "No text provided."}), 400

    text = clean_text(raw_text)

    try:
        if model_key in ("logistic_regression", "random_forest"):
            result = predict_classical(model_key, text)

        elif model_key in ("lstm", "bilstm"):
            result = predict_deep(model_key, text)

        elif model_key in ("bert", "roberta"):
            result = predict_transformer(model_key, text)

        else:
            return jsonify({"error": f"Unknown model '{model_key}'."}), 400

        result["model"] = model_key
        return jsonify(result)

    except FileNotFoundError as e:
        return jsonify({"error": f"Model artefact not found: {e}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Use port 5000 locally; Render/Hugging Face inject PORT env var
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
