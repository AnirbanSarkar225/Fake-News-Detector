import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load model
preprocessor = joblib.load("model/preprocessor.pkl")
model = joblib.load("model/fake_news_model.pkl")

# Load dataset
df = pd.read_csv("data/news.csv")

# Random sample of 5000 articles
df = df.sample(n=5000, random_state=42)

X = df["text"]
y_true = df["label"]

# Preprocess
X_processed = [preprocessor.preprocess_for_model(text) for text in X]

# Predict
y_pred = model.predict(X_processed)

print("\nAccuracy:")
print(accuracy_score(y_true, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred))