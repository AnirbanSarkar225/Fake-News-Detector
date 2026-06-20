import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load components
preprocessor = joblib.load("model/preprocessor.pkl")
model = joblib.load("model/fake_news_model.pkl")

# Load evaluation dataset
df = pd.read_csv("data/fake_news_evaluation_set.csv")

print(f"Loaded {len(df)} samples")

# Adjust column names if needed
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