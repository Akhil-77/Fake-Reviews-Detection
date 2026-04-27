"""
pattern_mining.py
=================
Mines class-conditional association rules from the itemsets produced by
feature_extraction.py, using both Apriori and FP-Growth.

Output: a JSON ruleset usable by llm_classifier.py and llm_generator.py.

Run AFTER feature_extraction.py:
    python pattern_mining.py --algo apriori --min_support 0.05 --min_confidence 0.6

Saved artefacts (./artifacts/):
    rules_apriori.json   or   rules_fpgrowth.json
"""

import argparse, json
import pandas as pd

from mlxtend.preprocessing       import TransactionEncoder
from mlxtend.frequent_patterns    import apriori, fpgrowth, association_rules


ARTIFACT_DIR = "artifacts"

LABEL_ITEMS = {"label_genuine", "label_suspicious"}


def load_itemsets(path: str) -> list[list[str]]:
    with open(path) as f:
        return json.load(f)


def mine(transactions, algo: str, min_support: float):
    te = TransactionEncoder()
    matrix = te.fit_transform(transactions)
    df = pd.DataFrame(matrix, columns=te.columns_)
    print(f"  Encoded {df.shape[0]:,} transactions × {df.shape[1]} unique items")

    if algo == "apriori":
        freq = apriori(df, min_support=min_support, use_colnames=True)
    elif algo == "fpgrowth":
        freq = fpgrowth(df, min_support=min_support, use_colnames=True)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    print(f"  Found {len(freq):,} frequent itemsets")
    return freq


def filter_class_rules(rules: pd.DataFrame, min_confidence: float, min_lift: float):
    """Keep only rules whose CONSEQUENT is exactly one label item."""
    def consequent_is_label(c):
        c = set(c)
        return len(c) == 1 and next(iter(c)) in LABEL_ITEMS

    out = rules[rules["consequents"].apply(consequent_is_label)]
    out = out[(out["confidence"] >= min_confidence) & (out["lift"] >= min_lift)]
    out = out.sort_values(["confidence", "lift", "support"], ascending=False)
    return out


def to_serialisable(rules: pd.DataFrame) -> list[dict]:
    out = []
    for _, r in rules.iterrows():
        out.append({
            "antecedents": sorted(list(r["antecedents"])),
            "consequent":  next(iter(r["consequents"])),
            "support":     round(float(r["support"]), 4),
            "confidence":  round(float(r["confidence"]), 4),
            "lift":        round(float(r["lift"]), 4),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["apriori", "fpgrowth"], default="apriori")
    ap.add_argument("--min_support",    type=float, default=0.05)
    ap.add_argument("--min_confidence", type=float, default=0.6)
    ap.add_argument("--min_lift",       type=float, default=1.1)
    ap.add_argument("--top_k",          type=int,   default=50)
    args = ap.parse_args()

    print(f"Loading {ARTIFACT_DIR}/itemsets_train.json …")
    transactions = load_itemsets(f"{ARTIFACT_DIR}/itemsets_train.json")

    print(f"Mining frequent itemsets ({args.algo}, min_support={args.min_support}) …")
    freq = mine(transactions, args.algo, args.min_support)

    print(f"Generating association rules (min_confidence={args.min_confidence}) …")
    rules = association_rules(freq, metric="confidence",
                              min_threshold=args.min_confidence)
    print(f"  {len(rules):,} raw rules")

    rules = filter_class_rules(rules, args.min_confidence, args.min_lift)
    print(f"  {len(rules):,} class-conditional rules after filtering")

    rules = rules.head(args.top_k)
    rules_out = to_serialisable(rules)

    out_path = f"{ARTIFACT_DIR}/rules_{args.algo}.json"
    with open(out_path, "w") as f:
        json.dump({
            "algo": args.algo,
            "min_support": args.min_support,
            "min_confidence": args.min_confidence,
            "min_lift": args.min_lift,
            "n_rules": len(rules_out),
            "rules": rules_out,
        }, f, indent=2)

    print(f"\n✅  Top {len(rules_out)} rules saved → {out_path}")
    if rules_out:
        print("\nTop 5 rules:")
        for r in rules_out[:5]:
            print(f"  {r['antecedents']} ⇒ {r['consequent']}   "
                  f"conf={r['confidence']:.2f}  lift={r['lift']:.2f}  "
                  f"sup={r['support']:.2f}")


if __name__ == "__main__":
    main()
