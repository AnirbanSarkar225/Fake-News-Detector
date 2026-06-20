"""
DistilBERT Fine-Tuning Script for Fake News Detection.

Fine-tunes distilbert-base-uncased on news.csv for binary classification (FAKE/REAL).
Saves the model to model/distilbert_fakenews/.

Usage:
    python scripts/train_bert.py
    python scripts/train_bert.py --epochs 5 --batch-size 16
"""

import os, sys, io, time, json, argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

def train_bert(epochs=3, batch_size=16, max_length=256, learning_rate=2e-5, max_samples=None):
    # Import here so the script can be imported without torch
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data_path = os.path.join(PROJECT_ROOT, 'data', 'news.csv')
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Standardize column names
    if 'label' not in df.columns:
        for col in df.columns:
            if col in ['class', 'target', 'category', 'fake']:
                df.rename(columns={col: 'label'}, inplace=True)
                break
    if 'text' not in df.columns:
        for col in df.columns:
            if col in ['content', 'article', 'body', 'news']:
                df.rename(columns={col: 'text'}, inplace=True)
                break
    
    df = df[['text', 'label']].dropna()
    
    # Standardize labels to 0/1
    label_map = {}
    for val in df['label'].unique():
        val_str = str(val).strip().upper()
        if val_str in ['FAKE', '0', 'FALSE', 'UNRELIABLE']:
            label_map[val] = 0
        else:
            label_map[val] = 1
    df['label'] = df['label'].map(label_map)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)
    
    print(f"Dataset size: {len(df):,} articles")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    # Split
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])
    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}")

    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Dataset class
    class NewsDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts.tolist()
            self.labels = labels.tolist()
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx])[:5000]  # Limit text length
            encoding = self.tokenizer(
                text, max_length=self.max_length, padding='max_length',
                truncation=True, return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)
            }

    train_dataset = NewsDataset(train_df['text'], train_df['label'], tokenizer, max_length)
    val_dataset = NewsDataset(val_df['text'], val_df['label'], tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps)

    # Training loop
    best_f1 = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"  Avg Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')
        print(f"  Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        print(classification_report(val_labels, val_preds, target_names=['FAKE', 'REAL']))

        if val_f1 > best_f1:
            best_f1 = val_f1
            # Save best model
            save_dir = os.path.join(PROJECT_ROOT, 'model', 'distilbert_fakenews')
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  ✓ Best model saved (F1: {best_f1:.4f})")

    # Save metrics
    metrics = {
        'accuracy': float(val_acc),
        'f1_score': float(best_f1),
        'confusion_matrix': confusion_matrix(val_labels, val_preds).tolist(),
        'epochs': epochs,
        'max_length': max_length,
        'batch_size': batch_size,
        'train_size': len(train_df),
        'val_size': len(val_df)
    }
    metrics_path = os.path.join(PROJECT_ROOT, 'model', 'distilbert_fakenews', 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Training complete! Best F1: {best_f1:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune DistilBERT for fake news detection')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max-samples', type=int, default=None, help='Limit dataset size for testing')
    args = parser.parse_args()
    train_bert(epochs=args.epochs, batch_size=args.batch_size, max_length=args.max_length, learning_rate=args.lr, max_samples=args.max_samples)
