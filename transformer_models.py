"""
transformer_models.py
=====================
Fine-tunes BERT and/or RoBERTa on the train/test CSVs produced by data_prep.py.

Run AFTER data_prep.py:
    python transformer_models.py

Saved artefacts (./artifacts/):
    bert_model/
    roberta_model/

Notes:
- First run needs internet access unless the Hugging Face base models are already cached.
- Full transformer training on CPU is very slow, so CPU defaults use a stratified subset.
  Set TRANSFORMER_TRAIN_LIMIT=0 and TRANSFORMER_TEST_LIMIT=0 to use all rows.
- To train only BERT:    TRANSFORMER_MODELS=bert python transformer_models.py
- To train only RoBERTa: TRANSFORMER_MODELS=roberta python transformer_models.py
"""

import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ---- config -----------------------------------------------------------------
ARTIFACT_DIR = "artifacts"
BERT_NAME = "bert-base-uncased"
ROBERTA_NAME = "roberta-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SEQ_LEN = int(os.environ.get("TRANSFORMER_MAX_SEQ_LEN", "128"))
BATCH_SIZE = int(os.environ.get("TRANSFORMER_BATCH_SIZE", "16" if DEVICE == "cuda" else "8"))
EPOCHS = int(os.environ.get("TRANSFORMER_EPOCHS", "1"))
LR = float(os.environ.get("TRANSFORMER_LR", "2e-5"))
RANDOM_STATE = 42
LOG_EVERY = int(os.environ.get("TRANSFORMER_LOG_EVERY", "50"))

DEFAULT_TRAIN_LIMIT = "0" if DEVICE == "cuda" else "10000"
DEFAULT_TEST_LIMIT = "0" if DEVICE == "cuda" else "2000"
TRAIN_LIMIT = int(os.environ.get("TRANSFORMER_TRAIN_LIMIT", DEFAULT_TRAIN_LIMIT))
TEST_LIMIT = int(os.environ.get("TRANSFORMER_TEST_LIMIT", DEFAULT_TEST_LIMIT))
MODELS_TO_TRAIN = [
    item.strip().lower()
    for item in os.environ.get("TRANSFORMER_MODELS", "bert,roberta").split(",")
    if item.strip()
]
# -----------------------------------------------------------------------------

MODEL_REGISTRY = {
    "bert": (BERT_NAME, "bert_model"),
    "roberta": (ROBERTA_NAME, "roberta_model"),
}

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(RANDOM_STATE)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run python data_prep.py first.")


def stratified_limit(df, limit, split_name):
    n_classes = df["binary_label"].nunique()
    if limit <= 0 or len(df) <= limit or len(df) - limit < n_classes:
        print(f"{split_name}: using all {len(df):,} rows")
        return df.reset_index(drop=True)

    if n_classes > 1:
        _, sample_df = train_test_split(
            df,
            test_size=limit,
            stratify=df["binary_label"],
            random_state=RANDOM_STATE,
        )
    else:
        sample_df = df.sample(n=limit, random_state=RANDOM_STATE)

    print(f"{split_name}: using stratified sample of {len(sample_df):,} out of {len(df):,} rows")
    return sample_df.reset_index(drop=True)


def load_splits():
    train_path = f"{ARTIFACT_DIR}/train.csv"
    test_path = f"{ARTIFACT_DIR}/test.csv"
    require_file(train_path)
    require_file(test_path)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    for name, df in (("train.csv", train_df), ("test.csv", test_df)):
        required_cols = {"text", "binary_label"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")

    train_df["clean_text"] = train_df["text"].apply(clean_text)
    test_df["clean_text"] = test_df["text"].apply(clean_text)

    train_df = stratified_limit(train_df, TRAIN_LIMIT, "Train")
    test_df = stratified_limit(test_df, TEST_LIMIT, "Test")

    print(f"Train labels: {dict(zip(*np.unique(train_df['binary_label'], return_counts=True)))}")
    print(f"Test labels : {dict(zip(*np.unique(test_df['binary_label'], return_counts=True)))}")
    return train_df, test_df


def make_loaders(train_df, test_df, tokenizer):
    train_ds = ReviewDataset(
        train_df["clean_text"].tolist(),
        train_df["binary_label"].tolist(),
        tokenizer,
        MAX_SEQ_LEN,
    )
    test_ds = ReviewDataset(
        test_df["clean_text"].tolist(),
        test_df["binary_label"].tolist(),
        tokenizer,
        MAX_SEQ_LEN,
    )
    pin_memory = DEVICE == "cuda"
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=pin_memory),
        DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin_memory),
    )


def train_epoch(model, loader, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)

        out = model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
            labels=batch["labels"].to(DEVICE),
        )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += out.loss.item()
        if LOG_EVERY > 0 and step % LOG_EVERY == 0:
            print(
                f"  epoch {epoch} step {step:,}/{len(loader):,} "
                f"loss={total_loss / step:.4f}",
                flush=True,
            )

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(name, model, loader):
    model.eval()
    all_labels = []
    all_probs = []

    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
        ).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(batch["labels"].detach().cpu().numpy().tolist())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    y_pred = (all_probs > 0.5).astype(int)

    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print(f"{'=' * 55}")
    print(f"  Accuracy : {accuracy_score(all_labels, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(all_labels, all_probs):.4f}")
    print("\n  Classification Report:")
    print(classification_report(all_labels, y_pred, target_names=["fake", "real"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(all_labels, y_pred))


def fine_tune(model_name, save_dir, train_df, test_df):
    print(f"\n{'#' * 60}")
    print(f"  Fine-tuning {model_name}")
    print(f"  Device: {DEVICE}")
    print(f"  max_len={MAX_SEQ_LEN}, batch_size={BATCH_SIZE}, epochs={EPOCHS}, lr={LR}")
    print(f"{'#' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    ).to(DEVICE)

    train_loader, test_loader = make_loaders(train_df, test_df, tokenizer)
    total_steps = len(train_loader) * EPOCHS
    if total_steps == 0:
        raise ValueError("No training batches were created. Check train.csv and batch size.")

    optimizer = AdamW(model.parameters(), lr=LR, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, epoch)
        print(f"  Epoch {epoch}/{EPOCHS} | train loss: {loss:.4f}")

    evaluate(model_name, model, test_loader)

    output_dir = f"{ARTIFACT_DIR}/{save_dir}"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Saved -> {output_dir}")


def transformer_artefact_exists(save_dir):
    """A trained HF model dir is valid if it contains a config.json."""
    full_dir = os.path.join(ARTIFACT_DIR, save_dir)
    return os.path.isdir(full_dir) and os.path.exists(os.path.join(full_dir, "config.json"))


FORCE_RETRAIN = os.environ.get("FORCE_RETRAIN", "0") == "1"


if __name__ == "__main__":
    train_df, test_df = load_splits()

    for model_key in MODELS_TO_TRAIN:
        if model_key not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown transformer model '{model_key}'. "
                f"Choose from {sorted(MODEL_REGISTRY)}."
            )
        model_name, save_dir = MODEL_REGISTRY[model_key]
        if transformer_artefact_exists(save_dir) and not FORCE_RETRAIN:
            print(f"\nSkipping {model_name} — {ARTIFACT_DIR}/{save_dir} already exists. "
                  f"Set FORCE_RETRAIN=1 to refit.")
            continue
        fine_tune(model_name, save_dir, train_df, test_df)

    print("\nDone. Transformer models fine-tuned and saved.")
