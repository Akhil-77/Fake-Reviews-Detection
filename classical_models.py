"""
classical_models.py
===================
Trains Logistic Regression and Random Forest on the TF-IDF features produced
by data_prep.py, evaluates them, and saves the fitted models.

Run AFTER data_prep.py:
    python classical_models.py

Useful options:
    python classical_models.py --model logistic
    python classical_models.py --model random_forest
    python classical_models.py --rf-sample-size 50000

Saved artefacts (./artifacts/):
    logistic_regression.pkl
    random_forest.pkl
"""

import argparse
import os
import pickle

import numpy as np
from scipy.sparse import load_npz

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# ---- config -----------------------------------------------------------------
ARTIFACT_DIR = "artifacts"
RANDOM_STATE = 42

# Random Forest on TF-IDF can be very slow if every tree is unlimited-depth.
# These defaults keep it practical for a laptop/student project while still
# producing a real Random Forest model. Set RF_TRAIN_SAMPLE_SIZE=0 to use all rows.
RF_TRAIN_SAMPLE_SIZE = int(os.environ.get("RF_TRAIN_SAMPLE_SIZE", "50000"))
RF_N_ESTIMATORS = int(os.environ.get("RF_N_ESTIMATORS", "80"))
RF_MAX_DEPTH_RAW = os.environ.get("RF_MAX_DEPTH", "45")
RF_MAX_DEPTH = None if RF_MAX_DEPTH_RAW.lower() == "none" else int(RF_MAX_DEPTH_RAW)
RF_MAX_SAMPLES = float(os.environ.get("RF_MAX_SAMPLES", "0.50"))
RF_MIN_SAMPLES_SPLIT = int(os.environ.get("RF_MIN_SAMPLES_SPLIT", "10"))
RF_MIN_SAMPLES_LEAF = int(os.environ.get("RF_MIN_SAMPLES_LEAF", "3"))
RF_VERBOSE = int(os.environ.get("RF_VERBOSE", "1"))
# -----------------------------------------------------------------------------


def load_data():
    required = [
        "X_train_tfidf.npz",
        "X_test_tfidf.npz",
        "y_train.npy",
        "y_test.npy",
    ]
    missing = [name for name in required if not os.path.exists(os.path.join(ARTIFACT_DIR, name))]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {ARTIFACT_DIR}/. Run python data_prep.py first."
        )

    X_train = load_npz(f"{ARTIFACT_DIR}/X_train_tfidf.npz")
    X_test = load_npz(f"{ARTIFACT_DIR}/X_test_tfidf.npz")
    y_train = np.load(f"{ARTIFACT_DIR}/y_train.npy")
    y_test = np.load(f"{ARTIFACT_DIR}/y_test.npy")

    print(f"Loaded TF-IDF train matrix: {X_train.shape[0]:,} rows x {X_train.shape[1]:,} features")
    print(f"Loaded TF-IDF test matrix : {X_test.shape[0]:,} rows x {X_test.shape[1]:,} features")
    print(f"Train labels: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    return X_train, X_test, y_train, y_test


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print(f"{'=' * 55}")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        print(f"  ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["fake", "real"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def train_logistic(X_train, y_train):
    print("\nTraining Logistic Regression ...")
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def maybe_sample_for_random_forest(X_train, y_train, sample_size):
    """Use a stratified subset for RF so training does not look infinite."""
    n_classes = len(np.unique(y_train))
    if (
        sample_size is None
        or sample_size <= 0
        or X_train.shape[0] <= sample_size
        or X_train.shape[0] - sample_size < n_classes
    ):
        print(f"Random Forest will use all {X_train.shape[0]:,} training rows.")
        return X_train, y_train

    print(
        f"Random Forest will use a stratified sample of {sample_size:,} "
        f"out of {X_train.shape[0]:,} training rows."
    )
    _, X_sample, _, y_sample = train_test_split(
        X_train,
        y_train,
        test_size=sample_size,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )
    return X_sample, y_sample


def train_random_forest(X_train, y_train, sample_size=RF_TRAIN_SAMPLE_SIZE):
    print("\nTraining Random Forest ...")
    X_rf, y_rf = maybe_sample_for_random_forest(X_train, y_train, sample_size)

    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features="sqrt",
        bootstrap=True,
        max_samples=RF_MAX_SAMPLES,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=RF_VERBOSE,
    )
    print(
        "Random Forest params: "
        f"n_estimators={RF_N_ESTIMATORS}, max_depth={RF_MAX_DEPTH}, "
        f"max_features='sqrt', max_samples={RF_MAX_SAMPLES}, "
        f"min_samples_split={RF_MIN_SAMPLES_SPLIT}, min_samples_leaf={RF_MIN_SAMPLES_LEAF}"
    )
    model.fit(X_rf, y_rf)
    return model


def save_model(model, filename):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    path = os.path.join(ARTIFACT_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved -> {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train classical fake-review models.")
    parser.add_argument(
        "--model",
        choices=["all", "logistic", "random_forest"],
        default="all",
        help="Choose which model to train. Default: all",
    )
    parser.add_argument(
        "--rf-sample-size",
        type=int,
        default=RF_TRAIN_SAMPLE_SIZE,
        help="Rows used to train Random Forest. Use 0 for full data. Default: 50000",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    X_train, X_test, y_train, y_test = load_data()

    if args.model in ("all", "logistic"):
        lr = train_logistic(X_train, y_train)
        evaluate("Logistic Regression", lr, X_test, y_test)
        save_model(lr, "logistic_regression.pkl")

    if args.model in ("all", "random_forest"):
        rf = train_random_forest(X_train, y_train, sample_size=args.rf_sample_size)
        evaluate("Random Forest", rf, X_test, y_test)
        save_model(rf, "random_forest.pkl")

    print("\nDone. Classical models trained and saved.")
