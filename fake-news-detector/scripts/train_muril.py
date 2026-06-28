"""
MuRIL Fine-Tuning Script for Multilingual Fake News Detection.

Fine-tunes google/muril-base-cased on news data for binary classification.
Supports mixed English + Indian language datasets.

Usage:
    python scripts/train_muril.py
    python scripts/train_muril.py --epochs 5 --batch-size 16
"""

import os, sys, io, time, json, argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

MODEL_NAME = "google/muril-base-cased"
SAVE_DIR = os.path.join(PROJECT_ROOT, "model", "muril_fakenews")


def load_data(max_samples=None):
    """Load and merge training datasets."""
    frames = []

    # Primary dataset
    csv_path = os.path.join(PROJECT_ROOT, "data", "news.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        if "text" in df.columns and "label" in df.columns:
            frames.append(df[["text", "label"]].dropna())
            print(f"  Loaded news.csv: {len(frames[-1]):,} articles")

    # Indian Fake News Dataset
    ifnd_path = os.path.join(PROJECT_ROOT, "data", "IFND_dataset.csv")
    if os.path.exists(ifnd_path):
        try:
            ifnd = pd.read_csv(ifnd_path)
            ifnd.columns = [c.strip().lower() for c in ifnd.columns]
            # Adapt column names
            text_col = next((c for c in ifnd.columns if c in ("text", "content", "article", "statement", "news")), None)
            label_col = next((c for c in ifnd.columns if c in ("label", "class", "target", "category", "fake")), None)
            if text_col and label_col:
                ifnd = ifnd.rename(columns={text_col: "text", label_col: "label"})
                ifnd = ifnd[["text", "label"]].dropna()
                frames.append(ifnd)
                print(f"  Loaded IFND_dataset.csv: {len(ifnd):,} articles")
        except Exception as e:
            print(f"  Warning: Could not load IFND dataset: {e}")

    if not frames:
        raise FileNotFoundError("No training data found in data/ directory")

    combined = pd.concat(frames, ignore_index=True)

    # Standardize labels to 0/1
    label_map = {}
    for val in combined["label"].unique():
        val_str = str(val).strip().upper()
        if val_str in ("FAKE", "0", "FALSE", "UNRELIABLE"):
            label_map[val] = 0
        else:
            label_map[val] = 1
    combined["label"] = combined["label"].map(label_map)
    combined = combined.dropna(subset=["label"])
    combined["label"] = combined["label"].astype(int)

    if max_samples:
        combined = combined.sample(n=min(max_samples, len(combined)), random_state=42)

    return combined


def train_muril(epochs=3, batch_size=16, max_length=256, learning_rate=2e-5, max_samples=None):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device: {device} (AMP: {use_amp})")

    # Load data
    print("\nLoading datasets...")
    df = load_data(max_samples)
    print(f"Total dataset: {len(df):,} articles")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])
    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}")

    # Tokenizer
    print(f"\nLoading MuRIL tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    class MultilingualNewsDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts.tolist()
            self.labels = labels.tolist()
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx])[:5000]
            encoding = self.tokenizer(
                text, max_length=self.max_length, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "label": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    train_dataset = MultilingualNewsDataset(train_df["text"], train_df["label"], tokenizer, max_length)
    val_dataset = MultilingualNewsDataset(val_df["text"], val_df["label"], tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    print(f"Loading MuRIL model from {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Training loop with early stopping
    best_f1 = 0
    patience = 2
    no_improve = 0
    val_acc = 0.0
    val_labels = []
    val_preds = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        print(f"\nEpoch {epoch + 1}/{epochs}")

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"  Avg Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average="weighted")
        print(f"  Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        print(classification_report(val_labels, val_preds, target_names=["FAKE", "REAL"]))

        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            os.makedirs(SAVE_DIR, exist_ok=True)
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"  \u2713 Best model saved (F1: {best_f1:.4f})")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    # Save metrics
    metrics = {
        "accuracy": float(val_acc),
        "f1_score": float(best_f1),
        "confusion_matrix": confusion_matrix(val_labels, val_preds).tolist(),
        "epochs_trained": epoch + 1,
        "max_length": max_length,
        "batch_size": batch_size,
        "model_name": MODEL_NAME,
        "train_size": len(train_df),
        "val_size": len(val_df),
    }
    metrics_path = os.path.join(SAVE_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n\u2705 Training complete! Best F1: {best_f1:.4f}")
    print(f"Model saved to: {SAVE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune MuRIL for fake news detection")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    train_muril(epochs=args.epochs, batch_size=args.batch_size,
                max_length=args.max_length, learning_rate=args.lr,
                max_samples=args.max_samples)
