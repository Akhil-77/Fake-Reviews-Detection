"""
yelpzip_to_csv.py
=================
Converts the raw YelpZIP release (metadata.txt + reviewContent.txt) into the
single yelpzip.csv file expected by data_prep.py.

YelpZIP raw format (from Speagle/Rayana 2015, KDD):
    metadata.txt        : tab-separated  user_id, prod_id, rating, label, date
                          label = -1 (filtered/suspicious)  or  1 (recommended/genuine)
    reviewContent.txt   : tab-separated  user_id, prod_id, date, review_text

We join on (user_id, prod_id, date) and emit:
    text, label (fake/real), rating, user_id, prod_id, date

Usage:
    python yelpzip_to_csv.py --metadata path/to/metadata.txt \
                             --content  path/to/reviewContent.txt \
                             --out      yelpzip.csv
"""

import argparse
import pandas as pd


def load_metadata(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=["user_id", "prod_id", "rating", "label", "date"],
        dtype={"user_id": str, "prod_id": str, "rating": float,
               "label": int, "date": str},
        engine="python", on_bad_lines="skip",
    )
    return df


def load_content(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="\t", header=None,
        names=["user_id", "prod_id", "date", "text"],
        dtype={"user_id": str, "prod_id": str, "date": str, "text": str},
        engine="python", on_bad_lines="skip",
        quoting=3,  # QUOTE_NONE — review text contains stray quotes
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="path to metadata.txt")
    ap.add_argument("--content",  required=True, help="path to reviewContent.txt")
    ap.add_argument("--out",      default="yelpzip.csv")
    args = ap.parse_args()

    print(f"Loading metadata from  {args.metadata}")
    meta = load_metadata(args.metadata)
    print(f"  {len(meta):,} rows")

    print(f"Loading content  from  {args.content}")
    content = load_content(args.content)
    print(f"  {len(content):,} rows")

    print("Joining on (user_id, prod_id, date) …")
    merged = meta.merge(content, on=["user_id", "prod_id", "date"], how="inner")
    print(f"  {len(merged):,} merged rows")

    # YelpZIP labels: -1 = filtered/fake/suspicious, 1 = recommended/real
    merged["label_str"] = merged["label"].map({-1: "fake", 1: "real"})
    merged = merged.dropna(subset=["text", "label_str"])
    merged = merged[merged["text"].str.len() > 0]

    out = merged[["text", "label_str", "rating", "user_id", "prod_id", "date"]]
    out = out.rename(columns={"label_str": "label"})

    print(f"\nLabel distribution:\n{out['label'].value_counts()}\n")
    print(f"Writing {len(out):,} rows → {args.out}")
    out.to_csv(args.out, index=True)
    print("Done.")


if __name__ == "__main__":
    main()
