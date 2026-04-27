"""
feature_extraction.py
=====================
Turns each review into a categorical *itemset* suitable for pattern mining
(Apriori / FP-Growth). Each review becomes a list of binary features like:

    {moderate_sentiment, detailed_experience, high_fluency,
     exclamation_heavy, rating_5, label_genuine}

The discretisation thresholds below are calibrated against typical Yelp
review distributions; tune `THRESHOLDS` after you've seen your dataset's
percentiles.

Run AFTER data_prep.py:
    python feature_extraction.py

Saved artefacts (./artifacts/):
    itemsets_train.json   : list[list[str]]  one itemset per training review
    itemsets_test.json    : list[list[str]]  one itemset per test review
    feature_schema.json   : human-readable dump of the feature definitions
"""

import os, re, json
import numpy as np
import pandas as pd

# ── lightweight, no-network NLP deps ──────────────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except ImportError:
    _vader = None

try:
    import textstat
except ImportError:
    textstat = None


ARTIFACT_DIR = "artifacts"

# Bucket boundaries — tune these after exploring your data.
THRESHOLDS = {
    "length": {"short": 50, "medium": 200},      # word count
    "sentiment": {"neg": -0.3, "pos": 0.3},      # vader compound
    "exclamations": {"none": 0, "heavy": 3},     # count of '!'
    "questions": {"none": 0, "heavy": 2},        # count of '?'
    "fluency_grade": {"low": 6, "high": 12},     # Flesch-Kincaid grade
    "type_token_ratio": {"low": 0.4, "high": 0.7},
    "caps_ratio": {"low": 0.02, "high": 0.10},   # uppercase chars / total chars
    "avg_word_len": {"low": 4.0, "high": 5.5},
}


def _length_bucket(text: str) -> str:
    n = len(text.split())
    if n < THRESHOLDS["length"]["short"]:
        return "length_short"
    if n < THRESHOLDS["length"]["medium"]:
        return "length_medium"
    return "length_long"


def _sentiment_bucket(text: str) -> str:
    if _vader is None:
        return "sentiment_unknown"
    s = _vader.polarity_scores(text)["compound"]
    if s < THRESHOLDS["sentiment"]["neg"]:
        return "negative_sentiment"
    if s > THRESHOLDS["sentiment"]["pos"]:
        return "positive_sentiment"
    return "moderate_sentiment"


def _punctuation_features(text: str) -> list[str]:
    feats = []
    excl = text.count("!")
    if excl >= THRESHOLDS["exclamations"]["heavy"]:
        feats.append("exclamation_heavy")
    elif excl == 0:
        feats.append("no_exclamations")

    qs = text.count("?")
    if qs >= THRESHOLDS["questions"]["heavy"]:
        feats.append("questions_heavy")

    if "..." in text or "…" in text:
        feats.append("ellipsis_present")
    return feats


def _fluency_features(text: str) -> list[str]:
    feats = []
    if textstat is None or not text.strip():
        return feats
    try:
        grade = textstat.flesch_kincaid_grade(text)
        if grade < THRESHOLDS["fluency_grade"]["low"]:
            feats.append("low_fluency")
        elif grade > THRESHOLDS["fluency_grade"]["high"]:
            feats.append("high_fluency")
        else:
            feats.append("medium_fluency")
    except Exception:
        pass
    return feats


def _lexical_diversity_features(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return []
    ttr = len(set(words)) / len(words)
    if ttr < THRESHOLDS["type_token_ratio"]["low"]:
        return ["low_diversity"]
    if ttr > THRESHOLDS["type_token_ratio"]["high"]:
        return ["high_diversity"]
    return ["medium_diversity"]


def _caps_features(text: str) -> list[str]:
    if not text:
        return []
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return []
    caps = sum(c.isupper() for c in letters) / len(letters)
    if caps > THRESHOLDS["caps_ratio"]["high"]:
        return ["caps_heavy"]
    return []


def _word_len_features(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    avg = float(np.mean([len(w) for w in words]))
    if avg < THRESHOLDS["avg_word_len"]["low"]:
        return ["short_words"]
    if avg > THRESHOLDS["avg_word_len"]["high"]:
        return ["long_words"]
    return []


def _detail_features(text: str) -> list[str]:
    feats = []
    n = len(text.split())
    if n > THRESHOLDS["length"]["medium"]:
        feats.append("detailed_experience")
    if any(p in text.lower() for p in [
        "first time", "second time", "last week", "last month", "recommend",
        "ordered", "menu", "waiter", "server", "table",
    ]):
        feats.append("specific_details")
    return feats


def _rating_bucket(rating) -> str:
    try:
        r = float(rating)
    except (TypeError, ValueError):
        return "rating_unknown"
    return f"rating_{int(round(r))}"


def review_to_itemset(text: str, rating=None, label: int | None = None) -> list[str]:
    """
    Build the itemset for one review. label optional (only present at training time).
    """
    text = "" if not isinstance(text, str) else text
    items: list[str] = []
    items.append(_length_bucket(text))
    items.append(_sentiment_bucket(text))
    items += _punctuation_features(text)
    items += _fluency_features(text)
    items += _lexical_diversity_features(text)
    items += _caps_features(text)
    items += _word_len_features(text)
    items += _detail_features(text)
    if rating is not None:
        items.append(_rating_bucket(rating))
    if label is not None:
        items.append("label_genuine" if int(label) == 1 else "label_suspicious")
    # de-dup while preserving order
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def main():
    train = pd.read_csv(f"{ARTIFACT_DIR}/train.csv")
    test  = pd.read_csv(f"{ARTIFACT_DIR}/test.csv")

    print(f"Building itemsets for {len(train):,} train reviews …")
    train_items = [
        review_to_itemset(r["text"], r.get("rating"), int(r["binary_label"]))
        for _, r in train.iterrows()
    ]
    print(f"Building itemsets for {len(test):,} test reviews (no label feature) …")
    test_items = [
        review_to_itemset(r["text"], r.get("rating"), label=None)
        for _, r in test.iterrows()
    ]

    with open(f"{ARTIFACT_DIR}/itemsets_train.json", "w") as f:
        json.dump(train_items, f)
    with open(f"{ARTIFACT_DIR}/itemsets_test.json", "w") as f:
        json.dump(test_items, f)

    schema = {
        "thresholds": THRESHOLDS,
        "items_per_review": "variable, typically 6-12",
        "label_items": ["label_genuine", "label_suspicious"],
        "vader_available": _vader is not None,
        "textstat_available": textstat is not None,
    }
    with open(f"{ARTIFACT_DIR}/feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    print("\n✅  Itemsets written.")
    print(f"   Train: {ARTIFACT_DIR}/itemsets_train.json")
    print(f"   Test:  {ARTIFACT_DIR}/itemsets_test.json")
    print(f"   Schema: {ARTIFACT_DIR}/feature_schema.json")


if __name__ == "__main__":
    main()
