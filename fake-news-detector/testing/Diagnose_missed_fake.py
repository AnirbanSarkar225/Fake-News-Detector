"""
TruthShield Diagnostic — Credibility Distribution of Misclassified FAKE Articles

Runs the same 500-article E2E sample as test_suite.py (random_state=99) and,
for every TRUE FAKE article the system got wrong, prints:
  - which credibility bucket it landed in
  - which signal(s) look the most "off" for that article (ml_score, nlp_score,
    redflag_count, satire_score, ai_score, temporal_penalty)

This tells us whether the miss-FAKE problem is a threshold-placement issue
(scores clustered just above 0.55) or a signal-strength issue (scores spread
across the whole 0.45-0.65 Uncertain band, meaning no single signal is firing
hard enough on real-world fake news).

Usage (from project root, same venv as test_suite.py):
    python testing\\diagnose_missed_fake.py
"""

import os
import sys
import warnings
from pathlib import Path
from collections import Counter

import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.preprocess import TextPreprocessor
from utils.source_engine import SourceEngine
from utils.clickbait_detector import ClickbaitDetector
from utils.ai_detector import AIContentDetector
from utils.claim_verifier import ClaimVerifier
from src.app import predict_article

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def bucket_for(cred):
    """Same 5-level buckets used by predict_article, plus finer sub-buckets
    inside the 0.45-0.65 Uncertain zone so we can see where misses cluster."""
    if cred >= 0.65:
        return "0.65+ (Likely Real / Highly Credible)"
    elif cred >= 0.60:
        return "0.60-0.65 (Uncertain, upper)"
    elif cred >= 0.55:
        return "0.55-0.60 (Uncertain, just above threshold)"
    elif cred >= 0.50:
        return "0.50-0.55 (Uncertain, just below threshold)"
    elif cred >= 0.45:
        return "0.45-0.50 (Uncertain, lower)"
    elif cred >= 0.20:
        return "0.20-0.45 (Likely Fake)"
    else:
        return "<0.20 (High Risk Misinformation)"


def main():
    print(f"{CYAN}{BOLD}Loading pipeline components...{RESET}")
    preprocessor = TextPreprocessor()
    source_engine = SourceEngine()
    clickbait_detect = ClickbaitDetector()
    ai_detect = AIContentDetector()
    claim_verifier = ClaimVerifier()

    model_path = ROOT / "model" / "fake_news_model.pkl"
    model = joblib.load(model_path)

    csv_path = ROOT / "data" / "news.csv"
    df = pd.read_csv(csv_path)
    sample_size = min(500, len(df))
    df = df.sample(n=sample_size, random_state=99)  # same seed as test_suite.py Section 3

    texts = df["text"].fillna("").tolist()
    y_true = df["label"].tolist()

    print(f"{GREEN}Running full pipeline on {sample_size} articles (random_state=99, matches test_suite.py)...{RESET}\n")

    missed_fake = []  # list of dicts with full diagnostic detail
    caught_fake = []  # for comparison baseline
    bucket_counts = Counter()

    for i, (text, label) in enumerate(zip(texts, y_true), 1):
        try:
            res = predict_article(
                text, model, preprocessor,
                clickbait_detect, ai_detect, claim_verifier, source_engine,
                None,
            )
        except Exception:
            continue

        if label != "FAKE":
            continue  # only interested in true-FAKE articles

        cred = res["credibility"]
        pred = res["prediction"]
        ind = res.get("indicators", {})

        record = {
            "idx": i,
            "credibility": cred,
            "ml_score": res.get("ml_score", 0.5),
            "nlp_score": res.get("nlp_score", 0.5),
            "ai_score": res.get("ai_score", 0.0),
            "satire_score": res.get("satire_score", 0.0),
            "redflag_count": ind.get("redflag_count", 0),
            "temporal_penalty": ind.get("temporal_penalty", 0.0),
            "category": res.get("category", ""),
            "stance": res.get("article_stance", "NEUTRAL"),
            "text_preview": text[:90].replace("\n", " "),
        }

        if pred != "FAKE":
            missed_fake.append(record)
            bucket_counts[bucket_for(cred)] += 1
        else:
            caught_fake.append(record)

    # ── Report ──
    print(f"{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  Missed FAKE articles: {len(missed_fake)} / {len(missed_fake) + len(caught_fake)} true-FAKE articles{RESET}")
    print(f"{'='*70}{RESET}\n")

    print(f"{BOLD}Credibility bucket distribution of MISSED fake articles:{RESET}")
    for bucket, count in sorted(bucket_counts.items(), key=lambda x: x[0]):
        pct = count / len(missed_fake) * 100 if missed_fake else 0
        bar = "█" * int(pct / 2)
        print(f"  {bucket:<45} {count:>4}  ({pct:5.1f}%)  {bar}")
    print()

    # Average signal values: missed vs caught, to see what differs
    def avg(records, key):
        vals = [r[key] for r in records]
        return sum(vals) / len(vals) if vals else 0.0

    print(f"{BOLD}Average signal values — MISSED fake vs CAUGHT fake:{RESET}")
    for key in ["credibility", "ml_score", "nlp_score", "ai_score", "satire_score", "redflag_count", "temporal_penalty"]:
        m = avg(missed_fake, key)
        c = avg(caught_fake, key)
        diff = m - c
        print(f"  {key:<20} missed={m:6.3f}   caught={c:6.3f}   diff={diff:+6.3f}")
    print()

    # How many missed-fake articles have ZERO signal firing at all
    # (redflag_count==0, satire_score==0, temporal_penalty==0) — these can only
    # be caught by ml_score/nlp_score, nothing else.
    no_signal = [r for r in missed_fake if r["redflag_count"] == 0 and r["satire_score"] == 0.0 and r["temporal_penalty"] == 0.0]
    print(f"{BOLD}Missed-fake articles with ZERO red-flag/satire/temporal signal: "
          f"{len(no_signal)} / {len(missed_fake)} ({len(no_signal)/len(missed_fake)*100 if missed_fake else 0:.1f}%){RESET}")
    print(f"{DIM}(These can only be caught by raising ml_score/nlp_score weight or threshold —")
    print(f" no red-flag pattern or satire/temporal fix will help them.){RESET}\n")

    # Show 10 example near-miss articles (closest to threshold) for manual inspection
    near_misses = sorted(missed_fake, key=lambda r: abs(r["credibility"] - 0.55))[:10]
    print(f"{BOLD}10 closest near-misses (credibility nearest to 0.55 threshold):{RESET}")
    for r in near_misses:
        print(f"  cred={r['credibility']:.3f}  ml={r['ml_score']:.2f}  nlp={r['nlp_score']:.2f}  "
              f"rf={r['redflag_count']}  satire={r['satire_score']:.2f}  temporal={r['temporal_penalty']:.2f}  "
              f"| {r['text_preview']}...")
    print()


if __name__ == "__main__":
    main()