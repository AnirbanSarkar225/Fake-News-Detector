"""
Real-time Incremental Update Script for Fake News Detector.

Fetches the latest claims from the Google Fact Check Tools API,
maps their textual ratings to FAKE/REAL labels, appends new unique claims
to data/news.csv, and incrementally updates the model's weights using partial_fit().

Runs continuously in the background checking for updates every 4 hours.
"""

import os
import sys
import io
import time
import joblib
import pandas as pd
import requests

# Fix Unicode output on Windows consoles (cp1252 can't print box-drawing/emoji)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "fake_news_model.pkl")
PREPROCESSOR_PATH = os.path.join(PROJECT_ROOT, "model", "preprocessor.pkl")
NEWS_CSV_PATH = os.path.join(PROJECT_ROOT, "data", "news.csv")

# Google Fact Check Tools API Endpoint (Uses standard query for latest claims)
# Replace 'YOUR_GOOGLE_API_KEY' with your actual free API key from Google Cloud Console.
API_KEY = "AIzaSyC27TYhEGCJaOJ6gwcM8fvaUundJBpbwls"
GOOGLE_FACTCHECK_URL = f"https://factchecktools.googleapis.com/v1alpha1/claims:search"

# Check interval: 30 minutes (1,800 seconds)
CHECK_INTERVAL_SECONDS = 1800

def map_rating_to_label(rating):
    """
    Maps Google's textual ratings (provided by different fact checking organizations)
    to binary classification categories FAKE / REAL.
    """
    rating = str(rating).lower().strip()
    fake_terms = ["false", "incorrect", "misleading", "fake", "pants on fire", "unproven", "inaccurate", "hoax", "distorted"]
    real_terms = ["true", "correct", "accurate", "mostly true", "verified"]
    
    if any(term in rating for term in fake_terms):
        return "FAKE"
    elif any(term in rating for term in real_terms):
        return "REAL"
        
    return None

def fetch_and_update_model():
    print(f"\n📡 [{time.strftime('%Y-%m-%d %H:%M:%S')}] Querying Google Fact Check API...")
    
    if not API_KEY or API_KEY.strip() == "" or API_KEY.startswith("YOUR_"):
        print("❌ Warning: API key has not been configured.")
        print("   Please get a free Google Fact Check Tools API Key from console.cloud.google.com")
        print("   and update API_KEY in this script.")
        return

    # Query both general news and global regions for fact-checked claims
    queries = ["news", "India", "world", "global", "USA", "Europe", "Asia", "Africa", "Americas", "Australia"]
    all_claims = []

    for query_term in queries:
        params = {
            "key": API_KEY,
            "pageSize": 100,
            "languageCode": "en",
            "query": query_term
        }

        try:
            response = requests.get(GOOGLE_FACTCHECK_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            claims = data.get("claims", [])
            all_claims.extend(claims)
            print(f"   ✓ Fetched {len(claims)} claims for query '{query_term}'")
        except Exception as e:
            print(f"   ⚠️ API call failed for query '{query_term}': {e}")

    new_texts = []
    new_labels = []
    
    # 1. Standardize and filter claims
    for claim in all_claims:
        text = claim.get("text")
        claim_reviews = claim.get("claimReview", [])
        if not claim_reviews or not text:
            continue
            
        raw_rating = claim_reviews[0].get("textualRating", "")
        label = map_rating_to_label(raw_rating)
        
        if label:
            new_texts.append(text.strip())
            new_labels.append(label)
            
    if len(new_texts) == 0:
        print("   No valid news claims identified in this cycle.")
        return

    # 2. De-duplicate against existing news.csv to only process BRAND NEW claims
    unique_texts = []
    unique_labels = []
    
    if os.path.exists(NEWS_CSV_PATH):
        try:
            existing_df = pd.read_csv(NEWS_CSV_PATH)
            # Create a set of existing texts for O(1) lookup
            existing_texts = set(existing_df['text'].str.strip().tolist())
            
            for t, l in zip(new_texts, new_labels):
                if t not in existing_texts:
                    unique_texts.append(t)
                    unique_labels.append(l)
        except Exception as e:
            print(f"⚠️ Error reading news.csv for de-duplication: {e}")
            unique_texts = new_texts
            unique_labels = new_labels
    else:
        unique_texts = new_texts
        unique_labels = new_labels

    print(f"   ✓ Extracted {len(new_texts)} matching claims.")
    print(f"   ✓ Found {len(unique_texts)} brand new unique claims.")
    
    if len(unique_texts) == 0:
        print("   No new unique claims to process in this cycle.")
        return

    # 3. Append new unique claims to data/news.csv
    try:
        new_data_df = pd.DataFrame({"text": unique_texts, "label": unique_labels})
        if os.path.exists(NEWS_CSV_PATH):
            new_data_df.to_csv(NEWS_CSV_PATH, mode='a', header=False, index=False, encoding='utf-8')
            print(f"   💾 Appended {len(unique_texts)} claims to data/news.csv")
        else:
            new_data_df.to_csv(NEWS_CSV_PATH, index=False, encoding='utf-8')
            print(f"   💾 Created data/news.csv and saved {len(unique_texts)} claims")
    except Exception as e:
        print(f"⚠️ Failed to save new claims to news.csv: {e}")

    # 4. Load the existing model pipeline for incremental training
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found at: {MODEL_PATH}")
        print("   Please run 'python train_model.py' to train the initial model first.")
        return
        
    print("📂 Loading model weights...")
    pipeline = joblib.load(MODEL_PATH)
    
    # Extract components from pipeline
    tfidf = pipeline.named_steps['tfidf']
    classifier = pipeline.named_steps['classifier']
    
    # Load preprocessor to clean raw text
    if os.path.exists(PREPROCESSOR_PATH):
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        cleaned_texts = [preprocessor.preprocess_for_model(t) for t in unique_texts]
    else:
        cleaned_texts = unique_texts
        
    # Transform new data using existing TF-IDF vocab
    X_new = tfidf.transform(cleaned_texts)
    y_new = unique_labels
    
    # Run incremental partial fit
    print("🧠 Incrementally training the classifier...")
    try:
        classifier.partial_fit(X_new, y_new, classes=['FAKE', 'REAL'])
        joblib.dump(pipeline, MODEL_PATH)
        print("💾 Model weights successfully updated in real-time!")
        
    except Exception as e:
        print(f"❌ Failed to incrementally train model: {e}")

def main():
    import socket
    # Ensure only one instance of the daemon runs at a time using a socket lock
    try:
        global lock_socket
        lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        lock_socket.bind(('127.0.0.1', 47291))
    except socket.error:
        print("⚠️ Another instance of the update daemon is already running. Exiting.")
        sys.exit(0)

    print("==================================================================")
    print("🛡️ Fake News Detector — Real-Time Background Update Daemon")
    print(f"   Checking Google Fact Check API every {CHECK_INTERVAL_SECONDS // 60} minutes")
    print("==================================================================")
    
    # Run once immediately on startup
    try:
        fetch_and_update_model()
    except Exception as e:
        print(f"⚠️ Error during startup fetch: {e}")
        
    # Infinite loop to run continuously in the background
    while True:
        try:
            print(f"\n💤 Sleeping for {CHECK_INTERVAL_SECONDS // 60} minutes before next check...")
            time.sleep(CHECK_INTERVAL_SECONDS)
            fetch_and_update_model()
        except KeyboardInterrupt:
            print("\n👋 Background daemon stopped by user. Exiting.")
            sys.exit(0)
        except Exception as e:
            print(f"⚠️ Error in background daemon loop: {e}")
            # If an error happens, wait 60 seconds before trying again to avoid rapid failure loops
            time.sleep(60)

if __name__ == "__main__":
    main()
