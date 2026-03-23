"""
deep_models.py
==============
Trains LSTM and BiLSTM models on the padded sequences produced by data_prep.py.

Run AFTER data_prep.py:
    python deep_models.py

Saved artefacts (./artifacts/):
    lstm_model/          (SavedModel format)
    bilstm_model/        (SavedModel format)
"""

import os, pickle
import numpy as np

import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models     import Model
from tensorflow.keras.layers     import (Embedding, LSTM, Bidirectional,
                                          Dense, Dropout, GlobalMaxPooling1D)
from tensorflow.keras.callbacks  import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)

from torch.optim import AdamW


# ── config ─────────────────────────────────────────────────────────────────────
ARTIFACT_DIR  = "artifacts"
MAX_FEATURES  = 30_000        # must match data_prep.py
MAX_SEQ_LEN   = 200           # must match data_prep.py
EMBED_DIM     = 128
LSTM_UNITS    = 128
DROPOUT       = 0.3
BATCH_SIZE    = 64
EPOCHS        = 1
RANDOM_STATE  = 42
# ───────────────────────────────────────────────────────────────────────────────

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def load_data():
    X_train = np.load(f"{ARTIFACT_DIR}/X_train_seq.npy")
    X_test  = np.load(f"{ARTIFACT_DIR}/X_test_seq.npy")
    y_train = np.load(f"{ARTIFACT_DIR}/y_train.npy")
    y_test  = np.load(f"{ARTIFACT_DIR}/y_test.npy")
    return X_train, X_test, y_train, y_test


def build_lstm():
    inp = Input(shape=(MAX_SEQ_LEN,), name="input")
    x   = Embedding(MAX_FEATURES, EMBED_DIM, name="embedding")(inp)
    x   = LSTM(LSTM_UNITS, return_sequences=True, name="lstm")(x)
    x   = GlobalMaxPooling1D(name="pool")(x)
    x   = Dropout(DROPOUT)(x)
    out = Dense(1, activation="sigmoid", name="output")(x)
    model = Model(inp, out, name="LSTM")
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_bilstm():
    inp = Input(shape=(MAX_SEQ_LEN,), name="input")
    x   = Embedding(MAX_FEATURES, EMBED_DIM, name="embedding")(inp)
    x   = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True), name="bilstm")(x)
    x   = GlobalMaxPooling1D(name="pool")(x)
    x   = Dropout(DROPOUT)(x)
    out = Dense(1, activation="sigmoid", name="output")(x)
    model = Model(inp, out, name="BiLSTM")
    model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def get_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]


def evaluate(name, model, X_test, y_test):
    y_prob = model.predict(X_test, batch_size=BATCH_SIZE).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["fake", "real"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    # ── LSTM ─────────────────────────────────────────────────────────────────
    print("\nTraining LSTM …")
    lstm = build_lstm()
    lstm.summary()
    lstm.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(),
    )
    evaluate("LSTM", lstm, X_test, y_test)
    lstm.save(f"{ARTIFACT_DIR}/lstm_model")
    print(f"  Saved → {ARTIFACT_DIR}/lstm_model")

    # ── BiLSTM ───────────────────────────────────────────────────────────────
    print("\nTraining BiLSTM …")
    bilstm = build_bilstm()
    bilstm.summary()
    bilstm.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks(),
    )
    evaluate("BiLSTM", bilstm, X_test, y_test)
    bilstm.save(f"{ARTIFACT_DIR}/bilstm_model")
    print(f"  Saved → {ARTIFACT_DIR}/bilstm_model")

    print("\n✅  Deep learning models trained and saved.")
