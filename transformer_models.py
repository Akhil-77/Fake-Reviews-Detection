"""
transformer_models.py
=====================
Fine-tunes BERT (bert-base-uncased) and RoBERTa (roberta-base) on the
train/test CSVs produced by data_prep.py.

Run AFTER data_prep.py:
    python transformer_models.py

Saved artefacts (./artifacts/):
    bert_model/      (HuggingFace SavedModel directory)
    roberta_model/   (HuggingFace SavedModel directory)

Note: Training transformers requires a GPU.  On CPU it will work but be slow.
"""

import os, re
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, accuracy_score,
                              roc_auc_score, confusion_matrix)

import torch
from torch.utils.data  import Dataset, DataLoader
from transformers       import (AutoTokenizer, AutoModelForSequenceClassification,
                                AdamW, get_linear_schedule_with_warmup)

# ── config ─────────────────────────────────────────────────────────────────────
ARTIFACT_DIR  = "artifacts"
BERT_NAME     = "bert-base-uncased"
ROBERTA_NAME  = "roberta-base"
MAX_SEQ_LEN   = 200          # consistent with data_prep.py
BATCH_SIZE    = 16
EPOCHS        = 3
LR            = 2e-5
RANDOM_STATE  = 42
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
# ───────────────────────────────────────────────────────────────────────────────

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ── same cleaning as data_prep.py ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

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
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_splits():
    train_df = pd.read_csv(f"{ARTIFACT_DIR}/train.csv")
    test_df  = pd.read_csv(f"{ARTIFACT_DIR}/test.csv")
    train_df["clean_text"] = train_df["text"].apply(clean_text)
    test_df["clean_text"]  = test_df["text"].apply(clean_text)
    return train_df, test_df


def make_loaders(train_df, test_df, tokenizer):
    train_ds = ReviewDataset(
        train_df["clean_text"].tolist(), train_df["binary_label"].tolist(),
        tokenizer, MAX_SEQ_LEN
    )
    test_ds = ReviewDataset(
        test_df["clean_text"].tolist(), test_df["binary_label"].tolist(),
        tokenizer, MAX_SEQ_LEN
    )
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(test_ds,  batch_size=BATCH_SIZE),
    )


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out  = model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
            labels=batch["labels"].to(DEVICE),
        )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += out.loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(name, model, loader):
    model.eval()
    all_labels, all_probs = [], []
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(DEVICE),
            attention_mask=batch["attention_mask"].to(DEVICE),
        ).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(batch["labels"].numpy())

    y_pred = (np.array(all_probs) > 0.5).astype(int)
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy : {accuracy_score(all_labels, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(all_labels, all_probs):.4f}")
    print("\n  Classification Report:")
    print(classification_report(all_labels, y_pred, target_names=["fake", "real"]))
    print("  Confusion Matrix:")
    print(confusion_matrix(all_labels, y_pred))


def fine_tune(model_name, save_dir, train_df, test_df):
    print(f"\n{'#'*60}")
    print(f"  Fine-tuning  {model_name}")
    print(f"  Device: {DEVICE}")
    print(f"{'#'*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(DEVICE)

    train_loader, test_loader = make_loaders(train_df, test_df, tokenizer)

    optimizer = AdamW(model.parameters(), lr=LR, eps=1e-8)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler)
        print(f"  Epoch {epoch}/{EPOCHS}  |  train loss: {loss:.4f}")

    evaluate(model_name, model, test_loader)

    os.makedirs(f"{ARTIFACT_DIR}/{save_dir}", exist_ok=True)
    model.save_pretrained(f"{ARTIFACT_DIR}/{save_dir}")
    tokenizer.save_pretrained(f"{ARTIFACT_DIR}/{save_dir}")
    print(f"  Saved → {ARTIFACT_DIR}/{save_dir}")


if __name__ == "__main__":
    train_df, test_df = load_splits()

    fine_tune(BERT_NAME,    "bert_model",    train_df, test_df)
    fine_tune(ROBERTA_NAME, "roberta_model", train_df, test_df)

    print("\n✅  Transformer models fine-tuned and saved.")
