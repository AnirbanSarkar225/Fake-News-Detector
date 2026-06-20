"""
Model Training Script for Fake News Detector.

Trains a TF-IDF + PassiveAggressiveClassifier pipeline incrementally using partial_fit.
This prevents MemoryError on resource-constrained environments.
"""

import os
import sys
import io
import time
import gc
import json
import joblib
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# Fix Unicode output on Windows consoles (cp1252 can't print box-drawing/emoji)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Resolve all paths relative to project root, not the scripts directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from utils.preprocess import TextPreprocessor


def train_model():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   🛡️  Fake News Detector — Incremental Training         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    data_dir = os.path.join(PROJECT_ROOT, "data")
    combined_path = os.path.join(data_dir, "news.csv")

    if not os.path.exists(combined_path):
        print("=" * 60)
        print("  ERROR: Dataset news.csv not found!")
        print("=" * 60)
        sys.exit(1)

    preprocessor = TextPreprocessor()

    print("📂 Step 1/5: Fitting TF-IDF Vectorizer on a sample subset...")
    # Load first 20k rows just to build the vocabulary
    sample_df = pd.read_csv(combined_path, nrows=20000)
    sample_df.columns = [col.strip().lower() for col in sample_df.columns]

    if 'label' not in sample_df.columns:
        for col in sample_df.columns:
            if col in ['class', 'target', 'category', 'fake']:
                sample_df.rename(columns={col: 'label'}, inplace=True)
                break

    if 'text' not in sample_df.columns:
        for col in sample_df.columns:
            if col in ['content', 'article', 'body', 'news']:
                sample_df.rename(columns={col: 'text'}, inplace=True)
                break

    sample_df = sample_df[['text', 'label']].dropna()
    print(f"   Pre-processing {len(sample_df):,} sample articles...")
    sample_df['processed_text'] = sample_df['text'].apply(preprocessor.preprocess_for_model)
    sample_df = sample_df[sample_df['processed_text'].str.len() > 10]

    tfidf = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 1),
        max_df=0.95,
        min_df=5,
        sublinear_tf=True,
    )
    tfidf.fit(sample_df['processed_text'])
    print(f"   ✓ Vocabulary built with {len(tfidf.vocabulary_)} features.")

    # Free memory of subset
    del sample_df
    gc.collect()

    print("\n🤖 Step 2/5: Initializing classifier...")
    classifier = PassiveAggressiveClassifier(
        max_iter=100,
        C=0.5,
        random_state=42,
    )

    print("\n🧠 Step 3/5: Training incrementally in batches...")
    start_time = time.time()
    
    chunk_size = 2000
    chunk_idx = 1
    total_processed = 0
    val_chunk = None

    # Read the dataset chunk-by-chunk to keep memory overhead tiny
    for chunk in pd.read_csv(combined_path, chunksize=chunk_size):
        chunk.columns = [col.strip().lower() for col in chunk.columns]
        
        if 'label' not in chunk.columns:
            for col in chunk.columns:
                if col in ['class', 'target', 'category', 'fake']:
                    chunk.rename(columns={col: 'label'}, inplace=True)
                    break

        if 'text' not in chunk.columns:
            for col in chunk.columns:
                if col in ['content', 'article', 'body', 'news']:
                    chunk.rename(columns={col: 'text'}, inplace=True)
                    break

        chunk = chunk[['text', 'label']].dropna()
        if len(chunk) == 0:
            continue

        # Standardize labels
        label_map = {}
        for val in chunk['label'].unique():
            val_str = str(val).strip().upper()
            if val_str in ['FAKE', '0', 'FALSE', 'UNRELIABLE']:
                label_map[val] = 'FAKE'
            else:
                label_map[val] = 'REAL'
        chunk['label'] = chunk['label'].map(label_map)
        chunk = chunk.dropna(subset=['label'])

        chunk['processed_text'] = chunk['text'].apply(preprocessor.preprocess_for_model)
        chunk = chunk[chunk['processed_text'].str.len() > 10]

        if len(chunk) == 0:
            continue

        # Reserve the first chunk as the validation set
        if val_chunk is None:
            val_chunk = chunk[['processed_text', 'label']].copy()
            print(f"     - Reserved first chunk of {len(val_chunk):,} articles for validation.")
            del chunk
            gc.collect()
            continue

        X_chunk = tfidf.transform(chunk['processed_text'])
        y_chunk = chunk['label'].tolist()

        # Incremental fitting
        classifier.partial_fit(X_chunk, y_chunk, classes=['FAKE', 'REAL'])
        total_processed += len(chunk)
        if chunk_idx % 5 == 0 or chunk_idx == 1:
            print(f"     - Trained batch {chunk_idx} (Processed: {total_processed:,} articles)")
        
        chunk_idx += 1
        del chunk, X_chunk, y_chunk
        gc.collect()

    elapsed = time.time() - start_time
    print(f"   ✓ Training completed in {elapsed:.1f}s")
    print()

    print("📈 Step 4/5: Evaluating model on the reserved validation set...")
    X_val = tfidf.transform(val_chunk['processed_text'])
    y_val = val_chunk['label'].tolist()

    y_pred = classifier.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    print(f"\n   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    print("   Classification Report:")
    print("   " + "-" * 55)
    report = classification_report(y_val, y_pred, target_names=['FAKE', 'REAL'])
    for line in report.split('\n'):
        print(f"   {line}")

    cm = confusion_matrix(y_val, y_pred, labels=['FAKE', 'REAL'])
    print(f"\n   Confusion Matrix:")
    print(f"   {'':>12} Pred FAKE  Pred REAL")
    print(f"   {'True FAKE':>12}   {cm[0][0]:>5}      {cm[0][1]:>5}")
    print(f"   {'True REAL':>12}   {cm[1][0]:>5}      {cm[1][1]:>5}")
    print()

    print("💾 Step 5/5: Saving model components...")
    # Wrap in a pipeline so the saved model has the same API
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('classifier', classifier)
    ])

    model_dir = os.path.join(PROJECT_ROOT, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "fake_news_model.pkl")

    joblib.dump(pipeline, model_path)
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   💾 Model saved to: {model_path} ({model_size:.1f} MB)")

    preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"   💾 Preprocessor saved to: {preprocessor_path}")

    # ── Save evaluation metrics to JSON for dynamic dashboard loading ──
    precision = precision_score(y_val, y_pred, pos_label='REAL')
    recall = recall_score(y_val, y_pred, pos_label='REAL')
    f1 = f1_score(y_val, y_pred, pos_label='REAL')

    # Generate ROC-like data using decision function
    try:
        decision_scores = classifier.decision_function(X_val)
        thresholds = np.linspace(decision_scores.min(), decision_scores.max(), 100)
        fpr_list = []
        tpr_list = []
        y_val_binary = (np.array(y_val) == 'REAL').astype(int)
        for thresh in thresholds:
            y_pred_thresh = (decision_scores >= thresh).astype(int)
            tp = ((y_pred_thresh == 1) & (y_val_binary == 1)).sum()
            fp = ((y_pred_thresh == 1) & (y_val_binary == 0)).sum()
            fn = ((y_pred_thresh == 0) & (y_val_binary == 1)).sum()
            tn = ((y_pred_thresh == 0) & (y_val_binary == 0)).sum()
            fpr_list.append(float(fp / max(fp + tn, 1)))
            tpr_list.append(float(tp / max(tp + fn, 1)))
        auc_score = float(np.abs(np.trapz(tpr_list, fpr_list)))
    except Exception:
        fpr_list = []
        tpr_list = []
        auc_score = 0.0

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "labels": ["FAKE", "REAL"],
        "train_size": int(total_processed),
        "test_size": int(len(val_chunk)),
        "total_articles": int(total_processed + len(val_chunk)),
        "roc_fpr": fpr_list,
        "roc_tpr": tpr_list,
        "roc_auc": auc_score
    }

    metrics_path = os.path.join(model_dir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"   📊 Evaluation metrics saved to: {metrics_path}")

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║              ✅ Training Complete!                       ║")
    print("║                                                          ║")
    print(f"║   Accuracy: {accuracy*100:.2f}%                                    ║")
    print("║                                                          ║")
    print("║   Next: Run 'streamlit run app.py' to start the app!    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    train_model()
