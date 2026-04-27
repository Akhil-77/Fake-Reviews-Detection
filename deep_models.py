"""
deep_models.py
==============
Trains LSTM and BiLSTM models on the padded sequences produced by data_prep.py.

Run AFTER data_prep.py:
    python deep_models.py

Saved artefacts (./artifacts/):
    lstm_model.keras
    bilstm_model.keras

These .keras filenames are intentional. TensorFlow/Keras 3 does not save/load
plain extensionless directories with model.save()/load_model() the same way older
Keras versions did.
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    LSTM,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

# ---- config -----------------------------------------------------------------
ARTIFACT_DIR = "artifacts"
MAX_FEATURES = 30_000       # must match data_prep.py
MAX_SEQ_LEN = 200           # must match data_prep.py
EMBED_DIM = int(os.environ.get("EMBED_DIM", "128"))
LSTM_UNITS = int(os.environ.get("LSTM_UNITS", "128"))
DROPOUT = float(os.environ.get("DROPOUT", "0.3"))
BATCH_SIZE = int(os.environ.get("DEEP_BATCH_SIZE", "64"))
EPOCHS = int(os.environ.get("DEEP_EPOCHS", "1"))
RANDOM_STATE = 42
# -----------------------------------------------------------------------------

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def load_data():
    required = [
        "X_train_seq.npy",
        "X_test_seq.npy",
        "y_train.npy",
        "y_test.npy",
    ]
    missing = [name for name in required if not os.path.exists(os.path.join(ARTIFACT_DIR, name))]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {ARTIFACT_DIR}/. Run python data_prep.py first."
        )

    X_train = np.load(f"{ARTIFACT_DIR}/X_train_seq.npy")
    X_test = np.load(f"{ARTIFACT_DIR}/X_test_seq.npy")
    y_train = np.load(f"{ARTIFACT_DIR}/y_train.npy")
    y_test = np.load(f"{ARTIFACT_DIR}/y_test.npy")

    print(f"Loaded sequence train matrix: {X_train.shape}")
    print(f"Loaded sequence test matrix : {X_test.shape}")
    print(f"Train labels: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    return X_train, X_test, y_train, y_test


def build_lstm():
    inp = Input(shape=(MAX_SEQ_LEN,), name="input")
    x = Embedding(MAX_FEATURES, EMBED_DIM, mask_zero=True, name="embedding")(inp)
    x = LSTM(LSTM_UNITS, return_sequences=True, name="lstm")(x)
    x = GlobalMaxPooling1D(name="pool")(x)
    x = Dropout(DROPOUT)(x)
    out = Dense(1, activation="sigmoid", name="output")(x)
    model = Model(inp, out, name="LSTM")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def build_bilstm():
    inp = Input(shape=(MAX_SEQ_LEN,), name="input")
    x = Embedding(MAX_FEATURES, EMBED_DIM, mask_zero=True, name="embedding")(inp)
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True), name="bilstm")(x)
    x = GlobalMaxPooling1D(name="pool")(x)
    x = Dropout(DROPOUT)(x)
    out = Dense(1, activation="sigmoid", name="output")(x)
    model = Model(inp, out, name="BiLSTM")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def get_callbacks():
    return [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6),
    ]


def get_class_weight(y_train):
    classes = np.unique(y_train)
    if len(classes) < 2:
        return None
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
    print(f"Class weights: {class_weight}")
    return class_weight


def evaluate(name, model, X_test, y_test):
    y_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print(f"{'=' * 55}")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["fake", "real"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def save_keras_model(model, filename):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    path = os.path.join(ARTIFACT_DIR, filename)
    model.save(path)
    print(f"  Saved -> {path}")


def keras_artefact_exists(base_name):
    """Match the same lookup pattern used by app.py."""
    for suffix in (".keras", ".h5", ""):
        path = os.path.join(ARTIFACT_DIR, f"{base_name}{suffix}")
        if os.path.exists(path):
            return True
    return False


FORCE_RETRAIN = os.environ.get("FORCE_RETRAIN", "0") == "1"


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    class_weight = get_class_weight(y_train)

    if keras_artefact_exists("lstm_model") and not FORCE_RETRAIN:
        print("\nSkipping LSTM — artefact already exists. Set FORCE_RETRAIN=1 to retrain.")
    else:
        print("\nTraining LSTM ...")
        lstm = build_lstm()
        lstm.summary()
        lstm.fit(
            X_train,
            y_train,
            validation_split=0.1,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=get_callbacks(),
            class_weight=class_weight,
            verbose=1,
        )
        evaluate("LSTM", lstm, X_test, y_test)
        save_keras_model(lstm, "lstm_model.keras")

    if keras_artefact_exists("bilstm_model") and not FORCE_RETRAIN:
        print("\nSkipping BiLSTM — artefact already exists. Set FORCE_RETRAIN=1 to retrain.")
    else:
        print("\nTraining BiLSTM ...")
        bilstm = build_bilstm()
        bilstm.summary()
        bilstm.fit(
            X_train,
            y_train,
            validation_split=0.1,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=get_callbacks(),
            class_weight=class_weight,
            verbose=1,
        )
        evaluate("BiLSTM", bilstm, X_test, y_test)
        save_keras_model(bilstm, "bilstm_model.keras")

    print("\nDone. Deep learning models trained and saved.")
