"""
data_prep.py
============
Loads the YelpZIP dataset, preprocesses text, performs train/test split,
and saves all artefacts needed by every downstream model file.

Saved artefacts (all go to  ./artifacts/ ):
  - train.csv / test.csv           : split data frames
  - tfidf_vectorizer.pkl           : fitted TF-IDF (used by classical models)
  - tokenizer.pkl                  : Keras tokenizer (used by LSTM / BiLSTM)
  - label_encoder.pkl              : maps  fake/real  →  0/1
  - X_train_tfidf.npz              : sparse TF-IDF matrix (train)
  - X_test_tfidf.npz               : sparse TF-IDF matrix (test)
  - X_train_seq.npy  /  X_test_seq.npy   : padded sequences (LSTM)
  - y_train.npy      /  y_test.npy       : binary labels
"""

import os, re, pickle
import numpy as np
import pandas as pd
from scipy.sparse import save_npz

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── configuration ──────────────────────────────────────────────────────────────
DATA_PATH      = "yelpzip.csv"          # ← path to your raw CSV
ARTIFACT_DIR   = "artifacts"
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
MAX_FEATURES   = 30_000                 # vocabulary cap (TF-IDF & Keras tokenizer)
MAX_SEQ_LEN    = 200                    # max tokens kept per review (LSTM)
TFIDF_NGRAMS   = (1, 2)                 # unigrams + bigrams → consistent across models
# ───────────────────────────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """Lower-case, remove punctuation/digits, collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df = df.dropna(subset=["text", "label"])
    df["clean_text"] = df["text"].apply(clean_text)
    # normalise label: 'fake' → 0, 'real' → 1  (or -1/1 → 0/1)
    if df["label"].dtype == object:
        df["binary_label"] = (df["label"].str.strip().str.lower() == "real").astype(int)
    else:
        df["binary_label"] = (df["label"] > 0).astype(int)
    return df


def make_artifacts(df: pd.DataFrame):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # ── train / test split ──────────────────────────────────────────────────
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["binary_label"]
    )
    train_df.to_csv(f"{ARTIFACT_DIR}/train.csv", index=False)
    test_df.to_csv(f"{ARTIFACT_DIR}/test.csv",  index=False)
    print(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")

    X_train_raw = train_df["clean_text"].tolist()
    X_test_raw  = test_df["clean_text"].tolist()
    y_train     = train_df["binary_label"].values
    y_test      = test_df["binary_label"].values

    np.save(f"{ARTIFACT_DIR}/y_train.npy", y_train)
    np.save(f"{ARTIFACT_DIR}/y_test.npy",  y_test)

    # ── TF-IDF (classical models) ────────────────────────────────────────────
    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=TFIDF_NGRAMS,
        sublinear_tf=True,
    )
    X_train_tfidf = tfidf.fit_transform(X_train_raw)
    X_test_tfidf  = tfidf.transform(X_test_raw)

    save_npz(f"{ARTIFACT_DIR}/X_train_tfidf.npz", X_train_tfidf)
    save_npz(f"{ARTIFACT_DIR}/X_test_tfidf.npz",  X_test_tfidf)
    with open(f"{ARTIFACT_DIR}/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    print("TF-IDF vectorizer saved.")

    # ── Keras tokenizer + padded sequences (LSTM / BiLSTM) ──────────────────
    tokenizer = Tokenizer(num_words=MAX_FEATURES, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_raw)

    X_train_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_train_raw),
        maxlen=MAX_SEQ_LEN, padding="post", truncating="post"
    )
    X_test_seq = pad_sequences(
        tokenizer.texts_to_sequences(X_test_raw),
        maxlen=MAX_SEQ_LEN, padding="post", truncating="post"
    )

    np.save(f"{ARTIFACT_DIR}/X_train_seq.npy", X_train_seq)
    np.save(f"{ARTIFACT_DIR}/X_test_seq.npy",  X_test_seq)
    with open(f"{ARTIFACT_DIR}/tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("Keras tokenizer + padded sequences saved.")

    # ── label encoder (for reference / transformer trainer) ─────────────────
    le = LabelEncoder()
    le.fit(y_train)
    with open(f"{ARTIFACT_DIR}/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print("Label encoder saved.")

    print("\n✅  All artefacts written to ./artifacts/")


if __name__ == "__main__":
    df = load_and_clean(DATA_PATH)
    print(f"Dataset loaded: {len(df):,} rows  |  label distribution:\n{df['binary_label'].value_counts()}\n")
    make_artifacts(df)
