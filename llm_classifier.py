"""
llm_classifier.py
=================
Learned-Ruleset-Augmented LLM classifier — the novel contribution.

Reads the mined ruleset from pattern_mining.py and prompts a hosted LLM
(via ASU Research Computing's OpenAI-compatible endpoint) to classify
each review using the rules. This is the Chain-of-Rule classifier
described in Phase 5 of the writeup.

Run AFTER pattern_mining.py:
    python llm_classifier.py --rules artifacts/rules_apriori.json \\
                             --n_eval 500

Saved artefacts (./artifacts/):
    llm_predictions.json        : per-review prediction + reasoning
    llm_results.json            : aggregated metrics
"""

import os, json, argparse, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, roc_auc_score,
                              precision_recall_fscore_support)

from feature_extraction import review_to_itemset


ARTIFACT_DIR = "artifacts"


# ── client ────────────────────────────────────────────────────────────────────
def make_client():
    load_dotenv()
    api_key  = os.environ["ASU_API_KEY"]
    base_url = os.environ.get("ASU_BASE_URL", "https://openai.rc.asu.edu/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


# ── prompt construction ──────────────────────────────────────────────────────
def format_rules(rules: list[dict], top_k: int = 30) -> str:
    """Format mined rules as a numbered list for the prompt."""
    lines = []
    for i, r in enumerate(rules[:top_k], 1):
        ant = " AND ".join(r["antecedents"])
        lines.append(
            f"{i:2d}. IF [{ant}] THEN {r['consequent']}  "
            f"(conf={r['confidence']:.2f}, lift={r['lift']:.2f})"
        )
    return "\n".join(lines)


SYSTEM_PROMPT = """You are a fake review detector for the YelpZip dataset. \
You classify a review as either GENUINE (Yelp-recommended) or SUSPICIOUS (Yelp-filtered).

You will be given:
  1. A set of association rules mined from labeled training data.
     Each rule has an antecedent (set of feature flags) and a consequent
     (label_genuine or label_suspicious), with confidence and lift.
  2. The review text and the feature flags extracted from it.

Apply the rules using Chain-of-Rule reasoning:
  - Identify which rule antecedents match the review's feature flags.
  - Weight matches by confidence × lift.
  - If no rules match strongly, fall back to your contextual understanding
    of the review's authenticity (overly generic, exaggerated, lacking
    specifics, suspiciously promotional, etc.).
  - Output your final classification.

Respond with STRICT JSON only, no preamble:
{"label": "fake"|"real", "confidence": 0.0-1.0, "matched_rules": [rule_numbers], "reasoning": "<brief>"}

Where:
  - "fake" = SUSPICIOUS (label_suspicious / Yelp-filtered)
  - "real" = GENUINE   (label_genuine   / Yelp-recommended)"""


def build_user_prompt(rules_text: str, review_text: str, features: list[str]) -> str:
    return (
        f"MINED RULES:\n{rules_text}\n\n"
        f"REVIEW:\n\"\"\"\n{review_text[:1500]}\n\"\"\"\n\n"
        f"EXTRACTED FEATURE FLAGS: {features}\n\n"
        f"Classify this review."
    )


# ── single-review classification ─────────────────────────────────────────────
def classify_one(client, model, rules_text, review_text, features,
                 max_retries=3) -> dict:
    user_prompt = build_user_prompt(rules_text, review_text, features)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            if attempt == max_retries - 1:
                return {"label": "real", "confidence": 0.5,
                        "matched_rules": [], "reasoning": f"ERROR: {e}"}
            time.sleep(2 ** attempt)


# ── batch evaluation ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules",      default=f"{ARTIFACT_DIR}/rules_apriori.json")
    ap.add_argument("--n_eval",     type=int, default=500,
                    help="how many test reviews to classify (cost control)")
    ap.add_argument("--top_rules",  type=int, default=30)
    ap.add_argument("--model",      default=None,
                    help="override ASU_MODEL from .env")
    ap.add_argument("--workers",    type=int, default=8,
                    help="concurrent API calls")
    ap.add_argument("--out",        default=f"{ARTIFACT_DIR}/llm_predictions.json")
    args = ap.parse_args()

    load_dotenv()
    model = args.model or os.environ.get("ASU_MODEL", "qwen3-30b-a3b-instruct-2507")
    print(f"Model: {model}")

    # ── load rules ─────────────────────────────────────────────────────────
    with open(args.rules) as f:
        ruleset = json.load(f)
    rules_text = format_rules(ruleset["rules"], top_k=args.top_rules)
    print(f"Loaded {len(ruleset['rules'])} rules; using top {args.top_rules} in prompt")

    # ── load test reviews ──────────────────────────────────────────────────
    test_df = pd.read_csv(f"{ARTIFACT_DIR}/test.csv")
    if args.n_eval and args.n_eval < len(test_df):
        test_df = test_df.sample(args.n_eval, random_state=42).reset_index(drop=True)
    print(f"Classifying {len(test_df):,} test reviews …")

    client = make_client()

    # ── concurrent calls ───────────────────────────────────────────────────
    def task(idx):
        row = test_df.iloc[idx]
        feats = review_to_itemset(row["text"], row.get("rating"), label=None)
        result = classify_one(client, model, rules_text, row["text"], feats)
        result["idx"]      = int(idx)
        result["true"]     = "real" if int(row["binary_label"]) == 1 else "fake"
        result["features"] = feats
        return result

    predictions = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(task, i): i for i in range(len(test_df))}
        for i, fut in enumerate(as_completed(futures), 1):
            predictions.append(fut.result())
            if i % 50 == 0:
                print(f"  {i}/{len(test_df)}")

    predictions.sort(key=lambda x: x["idx"])

    # ── metrics ────────────────────────────────────────────────────────────
    y_true = [1 if p["true"]  == "real" else 0 for p in predictions]
    y_pred = [1 if p["label"] == "real" else 0 for p in predictions]
    y_prob = [p["confidence"] if p["label"] == "real" else 1 - p["confidence"]
              for p in predictions]

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=0
    )

    print(f"\n{'='*55}")
    print(f"  Learned-Ruleset-Augmented LLM ({model})")
    print(f"{'='*55}")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  ROC-AUC      : {auc:.4f}")
    print(f"  Fake P/R/F1  : {p:.4f} / {r:.4f} / {f:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["fake", "real"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # ── save ───────────────────────────────────────────────────────────────
    with open(args.out, "w") as f:
        json.dump(predictions, f, indent=2)
    with open(f"{ARTIFACT_DIR}/llm_results.json", "w") as f:
        json.dump({
            "model": model, "n_eval": len(test_df),
            "accuracy": float(acc), "roc_auc": float(auc),
            "fake_precision": float(p), "fake_recall": float(r), "fake_f1": float(f),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }, f, indent=2)
    print(f"\n  Predictions → {args.out}")
    print(f"  Metrics     → {ARTIFACT_DIR}/llm_results.json")


if __name__ == "__main__":
    main()
