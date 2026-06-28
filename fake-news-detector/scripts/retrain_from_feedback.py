"""
Automated Retraining from User Feedback for TruthShield.

Queries the feedback/misclassification tables, merges corrections
into the training set, and retrains the sklearn ensemble model.

Usage:
    python scripts/retrain_from_feedback.py
    python scripts/retrain_from_feedback.py --min-feedback 5 --finetune-bert
    python scripts/retrain_from_feedback.py --dry-run
"""

import os, sys, io, time, json, argparse, shutil, sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, f1_score

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

DB_PATH = os.path.join(PROJECT_ROOT, "data", "truthshield.db")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "fake_news_model.pkl")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "news.csv")


def ensure_tables(conn):
    """Create misclassifications table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS misclassifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_hash TEXT,
            article_text TEXT,
            model_prediction TEXT,
            correct_label TEXT,
            reported_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            included_in_retrain BOOLEAN DEFAULT 0
        )
    """)
    conn.commit()


def get_feedback_corrections(conn):
    """Extract disagreement records from feedback table."""
    corrections = []
    try:
        cursor = conn.execute("""
            SELECT text, model_prediction, user_verdict, notes
            FROM feedback
            WHERE user_verdict != 'Agree with AI'
        """)
        for row in cursor.fetchall():
            text = row[0]
            model_pred = row[1]
            user_verdict = row[2]

            # Map user verdict to label
            if "fake" in user_verdict.lower() or "disagree" in user_verdict.lower():
                if model_pred == "REAL":
                    correct_label = "FAKE"
                else:
                    correct_label = "REAL"
            elif "real" in user_verdict.lower():
                correct_label = "REAL"
            else:
                correct_label = "FAKE" if model_pred == "REAL" else "REAL"

            corrections.append({"text": text, "label": correct_label, "source": "feedback"})
    except Exception as e:
        print(f"  Warning: Could not read feedback table: {e}")

    # Also check misclassifications table
    try:
        cursor = conn.execute("""
            SELECT article_text, correct_label
            FROM misclassifications
            WHERE included_in_retrain = 0 AND article_text IS NOT NULL
        """)
        for row in cursor.fetchall():
            corrections.append({"text": row[0], "label": row[1], "source": "misclassification"})
    except Exception:
        pass

    return corrections


def retrain_model(train_df, dry_run=False):
    """Retrain the sklearn ensemble model."""
    import joblib
    from utils.preprocess import TextPreprocessor

    preprocessor = TextPreprocessor()

    print("  Preprocessing text...")
    X = [preprocessor.preprocess_for_model(str(t)) for t in train_df["text"].tolist()]
    y = train_df["label"].tolist()

    if dry_run:
        print(f"  [DRY RUN] Would retrain on {len(X):,} samples")
        return None

    # Load existing model to get the pipeline structure
    old_model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

    # Retrain
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
    from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV

    print("  Building TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=75000, ngram_range=(1, 3),
                            sublinear_tf=True, min_df=2)

    svc = CalibratedClassifierCV(LinearSVC(max_iter=3000, C=1.0), cv=3)
    lr = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    pac = CalibratedClassifierCV(PassiveAggressiveClassifier(max_iter=1000, C=1.0), cv=3)
    et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    ensemble = VotingClassifier(
        estimators=[("svc", svc), ("lr", lr), ("pac", pac), ("et", et)],
        voting="soft",
    )

    pipeline = Pipeline([("tfidf", tfidf), ("clf", ensemble)])

    print("  Training ensemble...")
    pipeline.fit(X, y)

    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Retrain model from user feedback")
    parser.add_argument("--min-feedback", type=int, default=10,
                        help="Minimum disagreements before retraining")
    parser.add_argument("--finetune-bert", action="store_true",
                        help="Also fine-tune DistilBERT")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without retraining")
    args = parser.parse_args()

    print("=" * 50)
    print("TruthShield Model Retraining from Feedback")
    print("=" * 50)

    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    ensure_tables(conn)

    # Get feedback corrections
    print("\nCollecting feedback corrections...")
    corrections = get_feedback_corrections(conn)
    print(f"  Found {len(corrections)} correction(s)")

    if len(corrections) < args.min_feedback:
        print(f"  Need at least {args.min_feedback} corrections to retrain. Skipping.")
        conn.close()
        return

    # Load base training data
    print("\nLoading base training data...")
    base_df = pd.read_csv(DATA_PATH)
    base_df.columns = [c.strip().lower() for c in base_df.columns]
    base_df = base_df[["text", "label"]].dropna()
    print(f"  Base data: {len(base_df):,} articles")

    # Merge corrections
    corr_df = pd.DataFrame(corrections)
    corr_df = corr_df[["text", "label"]]
    combined = pd.concat([base_df, corr_df], ignore_index=True)
    print(f"  Combined data: {len(combined):,} articles (+{len(corr_df)} corrections)")

    # Retrain
    print("\nRetraining model...")
    import joblib
    new_model = retrain_model(combined, dry_run=args.dry_run)

    if new_model is not None:
        # Backup old model
        if os.path.exists(MODEL_PATH):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = MODEL_PATH.replace(".pkl", f"_backup_{timestamp}.pkl")
            shutil.copy2(MODEL_PATH, backup_path)
            print(f"  Old model backed up to: {backup_path}")

        # Save new model
        joblib.dump(new_model, MODEL_PATH)
        print(f"  New model saved to: {MODEL_PATH}")

        # Mark feedback as processed
        try:
            conn.execute("UPDATE misclassifications SET included_in_retrain = 1 WHERE included_in_retrain = 0")
            conn.commit()
        except Exception:
            pass

    # Optional: fine-tune BERT
    if args.finetune_bert and not args.dry_run:
        print("\nFine-tuning DistilBERT on corrected data...")
        try:
            from scripts.train_bert import train_bert
            # Save corrections as temp CSV for BERT training
            temp_csv = os.path.join(PROJECT_ROOT, "data", "feedback_corrections.csv")
            corr_df.to_csv(temp_csv, index=False)
            print("  Running DistilBERT fine-tuning (this may take a while)...")
            train_bert(epochs=2, batch_size=16, max_samples=len(corr_df) + 1000)
        except Exception as e:
            print(f"  Warning: BERT fine-tuning failed: {e}")

    conn.close()
    print(f"\n{'='*50}")
    print("Retraining complete!")


if __name__ == "__main__":
    main()
