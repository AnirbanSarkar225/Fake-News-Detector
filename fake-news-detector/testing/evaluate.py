from pathlib import Path
import sys
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# Load components
preprocessor = joblib.load(ROOT / "model" / "preprocessor.pkl")
model = joblib.load(ROOT / "model" / "fake_news_model.pkl")

# Load evaluation dataset
df = pd.read_csv(
    ROOT / "data" / "test data" / "fake_news_evaluation_set.csv"
)

print(f"Loaded {len(df)} samples")

X = df["text"]
y_true = df["label"]

# Preprocess text
X_processed = [preprocessor.preprocess_for_model(text) for text in X]

# Predict
y_pred = model.predict(X_processed)

# Metrics
print("\nAccuracy:")
print(accuracy_score(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred))