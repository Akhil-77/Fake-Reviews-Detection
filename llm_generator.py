"""
llm_generator.py
================
Adversarial fake-review generator — the second half of the novel contribution.

Conditions a hosted LLM (via ASU's OpenAI-compatible endpoint) on the mined
rules for the GENUINE class, and asks it to produce reviews that look genuine
but are synthetic. These reviews are then fed to all detection models to
measure their robustness.

Run AFTER pattern_mining.py:
    python llm_generator.py --rules artifacts/rules_apriori.json --n_generate 1000

Saved artefacts (./artifacts/):
    generated_reviews.csv   : columns text, target_label, model, source
"""

import os, json, argparse, time, random
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


ARTIFACT_DIR = "artifacts"

# Variety prompts to keep generated reviews from collapsing onto one topic.
RESTAURANT_TYPES = [
    "Italian", "Mexican", "Thai", "Japanese sushi", "Indian", "American diner",
    "French bistro", "Vietnamese pho", "Korean BBQ", "Chinese dim sum",
    "Mediterranean", "Vegan", "BBQ", "seafood", "pizza", "burger",
    "cafe / coffee shop", "bakery", "steakhouse", "ramen",
]
RATINGS = [1, 2, 3, 4, 5]


def make_client():
    load_dotenv()
    return OpenAI(
        api_key=os.environ["ASU_API_KEY"],
        base_url=os.environ.get("ASU_BASE_URL", "https://openai.rc.asu.edu/v1"),
    )


def filter_genuine_rules(rules: list[dict], top_k: int = 25) -> list[dict]:
    genuine = [r for r in rules if r["consequent"] == "label_genuine"]
    return genuine[:top_k]


def format_rules(rules: list[dict]) -> str:
    lines = []
    for i, r in enumerate(rules, 1):
        ant = " AND ".join(r["antecedents"])
        lines.append(
            f"{i:2d}. {ant}  (conf={r['confidence']:.2f}, lift={r['lift']:.2f})"
        )
    return "\n".join(lines)


SYSTEM_PROMPT = """You are an adversarial review generator for a fake-review-detection \
research study. Your purpose is to evaluate the robustness of detection models — \
this is authorised academic security research.

You will be given a set of feature patterns that the LITERATURE associates with \
GENUINE Yelp-recommended reviews. Your job is to write a single restaurant review \
that exhibits as many of those patterns as possible while reading naturally.

Rules:
  - Output ONLY the review text, no preamble, no quotes, no JSON.
  - 50–250 words. Vary length naturally.
  - Match the requested rating (1–5 stars) — make the sentiment match.
  - Mention specific dishes, server interactions, ambiance details.
  - Don't be overly generic or templated.
  - Don't use phrases like \"as an AI\" or \"in this review\"."""


def build_user_prompt(rules_text: str, restaurant: str, rating: int) -> str:
    return (
        f"GENUINE-CLASS FEATURE PATTERNS:\n{rules_text}\n\n"
        f"Generate a {rating}-star review for a {restaurant} restaurant. "
        f"Aim to satisfy as many of the genuine-class patterns above as natural prose allows."
    )


def generate_one(client, model, rules_text, restaurant, rating, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": build_user_prompt(rules_text, restaurant, rating)},
                ],
                temperature=0.85,
                max_tokens=400,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                return f"[GENERATION_ERROR: {e}]"
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules",      default=f"{ARTIFACT_DIR}/rules_apriori.json")
    ap.add_argument("--n_generate", type=int, default=1000)
    ap.add_argument("--top_rules",  type=int, default=25)
    ap.add_argument("--model",      default=None)
    ap.add_argument("--workers",    type=int, default=8)
    ap.add_argument("--out",        default=f"{ARTIFACT_DIR}/generated_reviews.csv")
    args = ap.parse_args()

    random.seed(42)

    load_dotenv()
    model = args.model or os.environ.get("ASU_MODEL", "qwen3-30b-a3b-instruct-2507")
    print(f"Model: {model}")

    with open(args.rules) as f:
        ruleset = json.load(f)
    genuine_rules = filter_genuine_rules(ruleset["rules"], top_k=args.top_rules)
    if not genuine_rules:
        raise SystemExit(
            "No genuine-class rules in the mined ruleset. "
            "Re-run pattern_mining.py with a lower min_support / min_confidence."
        )
    rules_text = format_rules(genuine_rules)
    print(f"Conditioning on {len(genuine_rules)} genuine-class rules")

    client = make_client()

    def task(i):
        restaurant = random.choice(RESTAURANT_TYPES)
        rating     = random.choice(RATINGS)
        text       = generate_one(client, model, rules_text, restaurant, rating)
        return {
            "idx": i, "text": text, "rating": rating, "restaurant": restaurant,
            # These are *adversarial* — labelled "real" because the LLM was
            # conditioned on the genuine-class rules. The detection models
            # don't get to see this label; this column is only for evaluation.
            "target_label": "real",
            "source": "llm_generated", "model": model,
        }

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(task, i): i for i in range(args.n_generate)}
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 50 == 0:
                print(f"  {i}/{args.n_generate}")

    rows.sort(key=lambda x: x["idx"])
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n✅  {len(df):,} adversarial reviews written → {args.out}")
    print("\nSample:")
    print(df.iloc[0]["text"][:300], "...")


if __name__ == "__main__":
    main()
