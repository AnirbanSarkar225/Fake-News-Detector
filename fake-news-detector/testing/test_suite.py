"""
🛡️ TruthShield Fake News Detector — Unified Test Suite & Stress Tester
Consolidates all heuristic debugs, unit cases, and model evaluation routines into a single script.
Runs a 10,000-article stress test to evaluate throughput, latency, and limits.
"""

import os
import sys
import io
import time
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Fix Unicode output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# ── 1. Environment Setup ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# Import official decision engine functions directly from app.py
from src.app import (
    predict_article,
    get_preprocessor,
    get_clickbait_detector,
    get_ai_detector,
    get_claim_verifier,
    get_source_engine
)

# ANSI Colors for terminal output
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def run_unit_tests(model, preprocessor, clickbait_detector, ai_detector, claim_verifier, source_engine):
    """Run verification against engineered edge cases using the production decision engine."""
    print(f"\n{CYAN}{BOLD}=== SECTION 1: UNIT & HEURISTIC TESTS ==={RESET}")
    
    test_cases = [
        {
            "name": "Normal Real News",
            "expected": "REAL",
            "text": "Central bank raises interest rates to address persistent inflation pressures across the economy."
        },
        {
            "name": "Normal Fake News",
            "expected": "FAKE",
            "text": "Scientists confirm that dinosaurs still live underground beneath major cities."
        },
        {
            "name": "Extreme Fabrication (Immunex Gland)",
            "expected": "FAKE",
            "text": "BREAKING: Scientists Discover New Organ Inside Human Body That Grants Immunity to All Diseases. "
                    "The Immunex Gland produces a unique enzyme that destroys all pathogens on contact. "
                    "activating this organ through a special diet of rare Himalayan herbs could eliminate "
                    "the need for all vaccines and pharmaceutical drugs. Several companies reportedly tried "
                    "to suppress the research. Critics have pointed out no other team has been able to verify it, "
                    "and the World Health Organization declined to comment."
        },
        {
            "name": "Scam Clickbait",
            "expected": "FAKE",
            "text": "Secret frequency hidden in FM radio broadcasts can instantly increase human IQ by 50 points."
        },
        {
            "name": "Fact-Check / Debunking Article (Should resolve to REAL)",
            "expected": "REAL",
            "text": "Fact-checkers say this viral claim about vaccines causing autism is false. Experts debunked it. "
                    "Snopes rated it as false. There is no evidence linking vaccines to autism according to WHO and CDC."
        },
        {
            "name": "Sensationalist Fake Health",
            "expected": "FAKE",
            "text": "Eating ice cream exclusively has been shown to reverse aging completely and cure diabetes."
        },
        {
            "name": "Standard Economy Reporting",
            "expected": "REAL",
            "text": "Economic report highlights gradual recovery in manufacturing activity and rising exports."
        }
    ]
    
    passed = 0
    for tc in test_cases:
        text = tc["text"]
        expected = tc["expected"]
        
        # Invoke the official decision engine logic
        res = predict_article(
            text, model, preprocessor, clickbait_detector, ai_detector,
            claim_verifier, source_engine, None
        )
        
        final_prediction = res["prediction"]
        credibility = res["credibility"]
        redflag_count = res["indicators"].get("redflag_count", 0)
        ml_score = res.get("ml_score", 0.5)
        nlp_score = res.get("nlp_score", 0.5)
        category = res["category"]
        
        status = GREEN + "PASS" + RESET if final_prediction == expected else RED + "FAIL" + RESET
        if final_prediction == expected:
            passed += 1
            
        print(f"[{status}] {tc['name']}")
        print(f"  - Verdict: {final_prediction} (Expected: {expected})")
        print(f"  - Scores:  ML={ml_score:.3f} | NLP={nlp_score:.3f} | Credibility={credibility:.3f} | Category={category}")
        print("-" * 50)
        
    print(f"{BOLD}Unit Test Summary: {passed}/{len(test_cases)} Passed ({passed/len(test_cases)*100:.1f}%){RESET}\n")
    return passed == len(test_cases)


def run_stress_test(model, preprocessor, sample_size=10000):
    """Run model predictions on a large dataset sample to evaluate performance and latency limits."""
    print(f"{CYAN}{BOLD}=== SECTION 2: STRESS & EXHAUSTION TEST ==={RESET}")
    
    csv_path = ROOT / "data" / "news.csv"
    if not csv_path.exists():
        print(f"{RED}Error: Dataset data/news.csv not found. Skip stress test.{RESET}")
        return
        
    print(f"Loading primary corpus: {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Adjust sample size if corpus is smaller
    sample_size = min(sample_size, len(df))
    print(f"Sampling {sample_size} articles for stress evaluation (Random state=42)...")
    df = df.sample(n=sample_size, random_state=42)
    
    X = df["text"].tolist()
    y_true = df["label"].tolist()
    
    print("\nStarting execution timing...")
    start_time = time.time()
    
    # Run full text preprocessing pipeline
    print("* Running preprocessor rules...")
    X_processed = [preprocessor.preprocess_for_model(text) for text in X]
    
    # Run ML predictions in bulk
    print("* Running classifier predictions...")
    y_pred = model.predict(X_processed)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    throughput = sample_size / total_time
    avg_latency = (total_time / sample_size) * 1000  # ms
    
    print(f"\n{GREEN}{BOLD}=== STRESS TEST REPORT ==={RESET}")
    print(f"Total Time Elapsed:    {total_time:.2f} seconds")
    print(f"Pipeline Throughput:   {throughput:.1f} articles/second")
    print(f"Average Latency:       {avg_latency:.2f} ms per article")
    
    # Compute accuracy scores
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy:      {BOLD}{accuracy*100:.2f}%{RESET}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


def main():
    print(f"{CYAN}{BOLD}+------------------------------------------------------+{RESET}")
    print(f"{CYAN}{BOLD}|          TruthShield Unified Test Suite              |{RESET}")
    print(f"{CYAN}{BOLD}+------------------------------------------------------+{RESET}")
    
    # Load model and preprocessor files
    try:
        preprocessor = get_preprocessor()
        
        # Resolve model path
        model_path = ROOT / "model" / "fake_news_model.pkl"
        if model_path.exists():
            model = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        clickbait_detector = get_clickbait_detector()
        ai_detector = get_ai_detector()
        claim_verifier = get_claim_verifier()
        source_engine = get_source_engine()
        
        print(f"{GREEN}Successfully loaded model and preprocessor configurations.{RESET}")
    except Exception as e:
        print(f"{RED}Fatal: Failed to load models: {e}{RESET}")
        sys.exit(1)
        
    # Section 1: Unit Tests (Calls production predict_article)
    run_unit_tests(model, preprocessor, clickbait_detector, ai_detector, claim_verifier, source_engine)
    
    # Section 2: Stress Test (10,000 articles)
    run_stress_test(model, preprocessor, sample_size=10000)


if __name__ == "__main__":
    main()
