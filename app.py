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
"""

import os
import pickle
import re

import numpy as np
from flask import Flask, jsonify, render_template, request

# Lazy-load heavy deps only when the relevant model is first requested.
_tfidf = None
_tokenizer = None
_models_cache = {}

ARTIFACT_DIR = "artifacts"
MAX_SEQ_LEN = 200
DEVICE = "cpu"  # keep CPU for local/free-tier inference

app = Flask(__name__)


# ---- text cleaning: same as data_prep.py ------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---- artefact loaders --------------------------------------------------------
def require_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def get_tfidf():
    global _tfidf
    if _tfidf is None:
        path = f"{ARTIFACT_DIR}/tfidf_vectorizer.pkl"
        require_path(path)
        with open(path, "rb") as f:
            _tfidf = pickle.load(f)
    return _tfidf


def get_keras_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        path = f"{ARTIFACT_DIR}/tokenizer.pkl"
        require_path(path)
        with open(path, "rb") as f:
            _tokenizer = pickle.load(f)
    return _tokenizer


def load_pickle_model(filename):
    path = f"{ARTIFACT_DIR}/{filename}"
    require_path(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def load_keras_model(base_name):
    """Load Keras 3 .keras files, with fallbacks for older saved names."""
    import tensorflow as tf

    candidates = [
        f"{ARTIFACT_DIR}/{base_name}.keras",
        f"{ARTIFACT_DIR}/{base_name}.h5",
        f"{ARTIFACT_DIR}/{base_name}",
    ]
    for path in candidates:
        if os.path.exists(path):
            return tf.keras.models.load_model(path)

    raise FileNotFoundError(
        f"Could not find any of these model files: {', '.join(candidates)}. "
        f"Run python deep_models.py first."
    )


def get_model(model_key: str):
    if model_key in _models_cache:
        return _models_cache[model_key]

    if model_key == "logistic_regression":
        model = load_pickle_model("logistic_regression.pkl")

    elif model_key == "random_forest":
        model = load_pickle_model("random_forest.pkl")

    elif model_key == "lstm":
        model = load_keras_model("lstm_model")

    elif model_key == "bilstm":
        model = load_keras_model("bilstm_model")

    elif model_key in ("bert", "roberta"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        dir_name = "bert_model" if model_key == "bert" else "roberta_model"
        model_dir = f"{ARTIFACT_DIR}/{dir_name}"
        require_path(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        transformer_model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
        transformer_model.eval()
        model = (tokenizer, transformer_model)

    else:
        raise ValueError(f"Unknown model: {model_key}")

    _models_cache[model_key] = model
    return model


# ---- prediction helpers ------------------------------------------------------
def predict_classical(model_key: str, text: str) -> dict:
    vec = get_tfidf().transform([text])
    model = get_model(model_key)
    prob = float(model.predict_proba(vec)[0, 1])
    label = "real" if prob > 0.5 else "fake"
    confidence = prob if label == "real" else 1.0 - prob
    return {"label": label, "confidence": round(float(confidence), 4)}


def predict_deep(model_key: str, text: str) -> dict:
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    tokenizer = get_keras_tokenizer()
    seq = pad_sequences(
        tokenizer.texts_to_sequences([text]),
        maxlen=MAX_SEQ_LEN,
        padding="post",
        truncating="post",
    )
    model = get_model(model_key)
    prob = float(model.predict(seq, verbose=0)[0, 0])
    label = "real" if prob > 0.5 else "fake"
    confidence = prob if label == "real" else 1.0 - prob
    return {"label": label, "confidence": round(float(confidence), 4)}


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
    idx = int(np.argmax(probs))
    label = "real" if idx == 1 else "fake"
    return {"label": label, "confidence": round(float(probs[idx]), 4)}


# ---- routes -----------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    raw_text = data.get("text", "").strip()
    model_key = data.get("model", "logistic_regression").strip().lower()

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

    except FileNotFoundError as exc:
        return jsonify({"error": f"Model artefact not found: {exc}"}), 500
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
