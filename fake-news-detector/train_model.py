"""
Model Training Script for Fake News Detector.

Trains a TF-IDF + PassiveAggressiveClassifier pipeline on
the Fake and Real News Dataset from Kaggle.

Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Usage:
    1. Download the dataset from Kaggle
    2. Place 'Fake.csv' and 'True.csv' in the data/ directory
       OR place a combined 'news.csv' with columns: text, label
    3. Run: python train_model.py
"""

import os
import sys
import io
import time

# Fix Unicode output on Windows consoles (cp1252 can't print box-drawing/emoji)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Resolve all paths relative to this script's directory, not the CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

sys.path.insert(0, SCRIPT_DIR)
from utils.preprocess import TextPreprocessor


def load_dataset(data_dir: str = None) -> pd.DataFrame:
    """
    Load the Fake and Real News Dataset.

    Supports two formats:
    1. Separate files: 'Fake.csv' and 'True.csv' (Kaggle format)
    2. Combined file: 'news.csv' with 'text' and 'label' columns

    Args:
        data_dir: Path to the data directory

    Returns:
        DataFrame with 'text' and 'label' columns
    """
    if data_dir is None:
        data_dir = os.path.join(SCRIPT_DIR, "data")
    combined_path = os.path.join(data_dir, "news.csv")
    fake_path = os.path.join(data_dir, "Fake.csv")
    true_path = os.path.join(data_dir, "True.csv")

    if os.path.exists(combined_path):
        print("[INFO] Loading combined dataset: news.csv")
        df = pd.read_csv(combined_path)

        df.columns = [col.strip().lower() for col in df.columns]

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

    elif os.path.exists(fake_path) and os.path.exists(true_path):
        print("[INFO] Loading separate Fake.csv and True.csv files")
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)

        fake_df['label'] = 'FAKE'
        true_df['label'] = 'REAL'

        df = pd.concat([fake_df, true_df], ignore_index=True)
        df.columns = [col.strip().lower() for col in df.columns]

        if 'title' in df.columns and 'text' in df.columns:
            df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        elif 'title' in df.columns:
            df.rename(columns={'title': 'text'}, inplace=True)

    else:
        print("=" * 60)
        print("  ERROR: Dataset not found!")
        print("=" * 60)
        print()
        print("  Please download the dataset from Kaggle:")
        print("  https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
        print()
        print("  Then place the files in the 'data/' directory:")
        print("    Option A: data/Fake.csv and data/True.csv")
        print("    Option B: data/news.csv (combined, with 'text' and 'label' columns)")
        print()
        print("=" * 60)
        sys.exit(1)

    if 'text' not in df.columns or 'label' not in df.columns:
        print(f"[ERROR] Dataset must have 'text' and 'label' columns.")
        print(f"        Found columns: {list(df.columns)}")
        sys.exit(1)

    return df[['text', 'label']].dropna()


def train_model():
    """
    Main training pipeline.

    Steps:
    1. Load and validate dataset
    2. Preprocess text data
    3. Train TF-IDF + PassiveAggressiveClassifier pipeline
    4. Evaluate on test set
    5. Save model to disk
    """
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          Fake News Detector — Model Training            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    print("📂 Step 1/5: Loading dataset...")
    df = load_dataset()
    print(f"   ✓ Loaded {len(df):,} articles")
    print(f"   ✓ Label distribution:")
    for label, count in df['label'].value_counts().items():
        print(f"     - {label}: {count:,} ({count/len(df)*100:.1f}%)")
    print()

    label_map = {}
    unique_labels = df['label'].unique()
    for lbl in unique_labels:
        lbl_str = str(lbl).strip().upper()
        if lbl_str in ['FAKE', '0', 'FALSE', 'UNRELIABLE']:
            label_map[lbl] = 'FAKE'
        else:
            label_map[lbl] = 'REAL'
    df['label'] = df['label'].map(label_map)

    print("🔧 Step 2/5: Preprocessing text...")
    preprocessor = TextPreprocessor()
    tqdm.pandas(desc="   Processing articles")
    df['processed_text'] = df['text'].progress_apply(preprocessor.preprocess_for_model)

    df = df[df['processed_text'].str.len() > 10]
    print(f"   ✓ {len(df):,} articles after preprocessing")
    print()

    print("📊 Step 3/5: Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'],
        df['label'],
        test_size=0.2,
        random_state=42,
        stratify=df['label']
    )
    print(f"   ✓ Training set: {len(X_train):,} articles")
    print(f"   ✓ Test set:     {len(X_test):,} articles")
    print()

    print("🤖 Step 4/5: Training model...")
    start_time = time.time()

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2,
            sublinear_tf=True,
        )),
        ('classifier', PassiveAggressiveClassifier(
            max_iter=100,
            C=0.5,
            random_state=42,
            class_weight='balanced',
        ))
    ])

    pipeline.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"   ✓ Training completed in {elapsed:.1f}s")
    print()

    print("📈 Step 5/5: Evaluating model...")
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    print("   Classification Report:")
    print("   " + "-" * 55)
    report = classification_report(y_test, y_pred, target_names=['FAKE', 'REAL'])
    for line in report.split('\n'):
        print(f"   {line}")

    cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
    print(f"\n   Confusion Matrix:")
    print(f"   {'':>12} Pred FAKE  Pred REAL")
    print(f"   {'True FAKE':>12}   {cm[0][0]:>5}      {cm[0][1]:>5}")
    print(f"   {'True REAL':>12}   {cm[1][0]:>5}      {cm[1][1]:>5}")
    print()

    model_dir = os.path.join(SCRIPT_DIR, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "fake_news_model.pkl")

    joblib.dump(pipeline, model_path)
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   💾 Model saved to: {model_path} ({model_size:.1f} MB)")

    preprocessor_path = os.path.join(model_dir, "preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"   💾 Preprocessor saved to: {preprocessor_path}")

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
