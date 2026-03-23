"""
classical_models.py
===================
Trains Logistic Regression and Random Forest on the TF-IDF features
produced by data_prep.py, evaluates them, and saves the fitted models.

Run AFTER data_prep.py:
    python classical_models.py

Saved artefacts (./artifacts/):
    logistic_regression.pkl
    random_forest.pkl
"""

import pickle, os
import numpy as np
from scipy.sparse import load_npz

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.metrics        import (classification_report, confusion_matrix,
                                    accuracy_score, roc_auc_score)

# ── config ─────────────────────────────────────────────────────────────────────
ARTIFACT_DIR = "artifacts"
RANDOM_STATE = 42
# ───────────────────────────────────────────────────────────────────────────────


def load_data():
    X_train = load_npz(f"{ARTIFACT_DIR}/X_train_tfidf.npz")
    X_test  = load_npz(f"{ARTIFACT_DIR}/X_test_tfidf.npz")
    y_train = np.load(f"{ARTIFACT_DIR}/y_train.npy")
    y_test  = np.load(f"{ARTIFACT_DIR}/y_test.npy")
    return X_train, X_test, y_train, y_test


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["fake", "real"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def train_logistic(X_train, y_train):
    print("Training Logistic Regression …")
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


def train_random_forest(X_train, y_train):
    print("Training Random Forest …")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def save_model(model, filename):
    path = os.path.join(ARTIFACT_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved → {path}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    # ── Logistic Regression ──────────────────────────────────────────────────
    lr = train_logistic(X_train, y_train)
    evaluate("Logistic Regression", lr, X_test, y_test)
    save_model(lr, "logistic_regression.pkl")

    # ── Random Forest ────────────────────────────────────────────────────────
    rf = train_random_forest(X_train, y_train)
    evaluate("Random Forest", rf, X_test, y_test)
    save_model(rf, "random_forest.pkl")

    print("\n✅  Classical models trained and saved.")
