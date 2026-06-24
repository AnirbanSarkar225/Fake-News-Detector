"""
Model Training Script for Fake News Detector.

Trains an Ensemble (LinearSVC + LogisticRegression + PassiveAggressive + ExtraTrees) pipeline
with TF-IDF (75k features, trigrams) for robust fake news classification.

Ensemble uses soft voting via CalibratedClassifierCV for probability calibration.
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
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
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   🛡️  TruthShield — Production Ensemble Training            ║")
    print("║   LinearSVC + LogisticRegression + PA + ExtraTrees          ║")
    print("║   TF-IDF: 75k features, (1,3) n-grams                      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    data_dir = os.path.join(PROJECT_ROOT, "data")
    combined_path = os.path.join(data_dir, "news.csv")

    if not os.path.exists(combined_path):
        print("=" * 60)
        print("  ERROR: Dataset news.csv not found!")
        print("=" * 60)
        sys.exit(1)

    preprocessor = TextPreprocessor()

    # ── Step 1: Load and prepare data ──────────────────────────────
    print("📂 Step 1/6: Loading dataset...")
    df = pd.read_csv(combined_path, low_memory=False)
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

    df = df[['text', 'label']].dropna()
    print(f"   Loaded {len(df):,} articles.")

    # Standardize labels
    label_map = {}
    for val in df['label'].unique():
        val_str = str(val).strip().upper()
        if val_str in ['FAKE', '0', 'FALSE', 'UNRELIABLE']:
            label_map[val] = 'FAKE'
        else:
            label_map[val] = 'REAL'
    df['label'] = df['label'].map(label_map)
    df = df.dropna(subset=['label'])

    # Continuous Learning: Load feedback from SQLite database if available
    db_path = os.path.join(PROJECT_ROOT, "data", "truthshield.db")
    if os.path.exists(db_path):
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            feedback_df = pd.read_sql_query(
                "SELECT text, model_prediction, user_verdict FROM feedback WHERE user_verdict IN ('Agree with AI', 'Disagree with AI')",
                conn
            )
            conn.close()
            if not feedback_df.empty:
                print(f"   Found {len(feedback_df)} user feedback entries in database for retraining.")
                feedback_data = []
                for _, row in feedback_df.iterrows():
                    text = row['text']
                    pred = str(row['model_prediction']).strip().upper()
                    verdict = str(row['user_verdict']).strip()
                    
                    if verdict == 'Agree with AI':
                        label = pred
                    elif verdict == 'Disagree with AI':
                        label = 'REAL' if pred == 'FAKE' else 'FAKE'
                    else:
                        continue
                    feedback_data.append({'text': text, 'label': label})
                
                if feedback_data:
                    feedback_cleaned = pd.DataFrame(feedback_data)
                    print(f"   ✓ Integrating {len(feedback_cleaned)} corrected training examples from user feedback loop.")
                    df = pd.concat([df, feedback_cleaned], ignore_index=True)
        except Exception as e:
            print(f"   ⚠️ Could not load feedback database for continuous learning: {e}")

    label_counts = df['label'].value_counts()
    print(f"   Label distribution: {label_counts.to_dict()}")

    # ── Step 2: Preprocess text ────────────────────────────────────
    print("\n🔧 Step 2/6: Preprocessing text...")
    start_time = time.time()
    df['processed_text'] = df['text'].apply(preprocessor.preprocess_for_model)
    df = df[df['processed_text'].str.len() > 10]
    print(f"   ✓ Preprocessed {len(df):,} articles in {time.time()-start_time:.1f}s")

    # ── Step 3: Train/Test split ───────────────────────────────────
    print("\n📊 Step 3/6: Splitting data (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'],
        test_size=0.20, random_state=42, stratify=df['label']
    )
    print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Free memory
    del df
    gc.collect()

    # ── Step 4: TF-IDF Vectorization ──────────────────────────────
    print("\n📐 Step 4/6: Fitting TF-IDF Vectorizer (75k features, trigrams)...")
    tfidf = TfidfVectorizer(
        max_features=75000,
        ngram_range=(1, 3),  # Trigrams for better context
        max_df=0.90,
        min_df=3,            # Allow rarer but discriminative terms
        sublinear_tf=True,
    )
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"   ✓ Vocabulary: {len(tfidf.vocabulary_):,} features")
    print(f"   ✓ Train matrix: {X_train_tfidf.shape}")

    # ── Step 5: Train Ensemble ─────────────────────────────────────
    print("\n🤖 Step 5/6: Training ensemble classifier...")
    start_time = time.time()

    # Build soft-voting ensemble
    print("\n   Building VotingClassifier (soft voting)...")

    # Since VotingClassifier doesn't support pre-fitted estimators easily,
    # we create a fresh ensemble and train it
    from sklearn.ensemble import ExtraTreesClassifier
    
    ensemble = VotingClassifier(
        estimators=[
            ('svc', CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=5000, random_state=42, verbose=1), cv=3, method='sigmoid')),
            ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='lbfgs', verbose=1)),
            ('pa', CalibratedClassifierCV(PassiveAggressiveClassifier(C=1.0, max_iter=500, random_state=42, verbose=1), cv=3, method='sigmoid')),
            ('et', ExtraTreesClassifier(n_estimators=100, max_depth=25, random_state=42, n_jobs=-1, verbose=1)),
        ],
        voting='soft',
        weights=[3, 2, 2, 1],  # Rebalanced weights for optimal ensemble accuracy
    )
    ensemble.fit(X_train_tfidf, y_train)

    elapsed = time.time() - start_time
    print(f"   ✓ Ensemble training completed in {elapsed:.1f}s")

    # ── Step 6: Evaluate ───────────────────────────────────────────
    print("\n📈 Step 6/6: Evaluating ensemble on test set...")
    y_pred = ensemble.predict(X_test_tfidf)
    y_proba = ensemble.predict_proba(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n   Ensemble Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
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

    # Calculate individual classifier accuracies from already fitted ensemble estimators
    classes = ensemble.classes_
    def get_est_pred(est):
        pred = est.predict(X_test_tfidf)
        if len(pred) > 0 and not isinstance(pred[0], str):
            pred = np.array([classes[int(round(x))] for x in pred])
        return pred

    svc_acc = accuracy_score(y_test, get_est_pred(ensemble.estimators_[0]))
    lr_acc = accuracy_score(y_test, get_est_pred(ensemble.estimators_[1]))
    pa_acc = accuracy_score(y_test, get_est_pred(ensemble.estimators_[2]))
    et_acc = accuracy_score(y_test, get_est_pred(ensemble.estimators_[3]))

    # Comparison table
    print("   Individual Classifier Accuracy:")
    print(f"   {'LinearSVC':>22}: {svc_acc*100:.2f}%")
    print(f"   {'LogisticRegression':>22}: {lr_acc*100:.2f}%")
    print(f"   {'PassiveAggressive':>22}: {pa_acc*100:.2f}%")
    print(f"   {'ExtraTrees':>22}: {et_acc*100:.2f}%")
    print(f"   {'Ensemble (weighted)':>22}: {accuracy*100:.2f}%")
    print()

    # ── Save model ─────────────────────────────────────────────────
    print("💾 Saving model components...")

    # Wrap TF-IDF + Ensemble in a pipeline
    pipeline = Pipeline([
        ('tfidf', tfidf),
        ('classifier', ensemble)
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

    # ── Save evaluation metrics ────────────────────────────────────
    precision = precision_score(y_test, y_pred, pos_label='REAL')
    recall_val = recall_score(y_test, y_pred, pos_label='REAL')
    f1 = f1_score(y_test, y_pred, pos_label='REAL')

    # ROC curve data
    try:
        # Get probabilities for the REAL class
        classes = list(ensemble.classes_)
        real_idx = classes.index('REAL')
        y_test_binary = (np.array(y_test.tolist()) == 'REAL').astype(int)
        fpr_arr, tpr_arr, _ = roc_curve(y_test_binary, y_proba[:, real_idx])
        auc_score = float(auc(fpr_arr, tpr_arr))
        fpr_list = fpr_arr.tolist()
        tpr_list = tpr_arr.tolist()
        # Downsample for storage if too many points
        if len(fpr_list) > 200:
            indices = np.linspace(0, len(fpr_list)-1, 200, dtype=int)
            fpr_list = [fpr_list[i] for i in indices]
            tpr_list = [tpr_list[i] for i in indices]
    except Exception as e:
        print(f"   ⚠️ ROC computation failed: {e}")
        fpr_list = []
        tpr_list = []
        auc_score = 0.0

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall_val),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "labels": ["FAKE", "REAL"],
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "total_articles": int(len(X_train) + len(X_test)),
        "roc_fpr": fpr_list,
        "roc_tpr": tpr_list,
        "roc_auc": auc_score,
        "model_type": "Ensemble (LinearSVC + LogisticRegression + PassiveAggressive + ExtraTrees)",
        "tfidf_features": int(len(tfidf.vocabulary_)),
        "ngram_range": [1, 3],
        "individual_accuracy": {
            "LinearSVC": float(svc_acc),
            "LogisticRegression": float(lr_acc),
            "PassiveAggressive": float(pa_acc),
            "ExtraTrees": float(et_acc),
        },
    }

    metrics_path = os.path.join(model_dir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"   📊 Evaluation metrics saved to: {metrics_path}")

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              ✅ Training Complete!                           ║")
    print("║                                                              ║")
    print(f"║   Ensemble Accuracy: {accuracy*100:.2f}%                              ║")
    print(f"║   ROC-AUC: {auc_score:.4f}                                        ║")
    print("║                                                              ║")
    print("║   Models: TF-IDF Ensemble + Preprocessor                    ║")
    print("║   Next: Run 'python start.py' to launch the app!            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    train_model()
