"""
TruthShield Fake News Detector — Unified Test Suite & Stress Tester

Three sections:
  1. Unit tests     — 14 edge cases through the full predict_article pipeline
  2. ML bench       — Raw classifier accuracy/throughput at 10,000 articles
  3. E2E stress     — Full predict_article pipeline accuracy at configurable scale
"""

import os
import sys
import io
import time
import math
import warnings
from pathlib import Path

import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── Unicode fix for Windows consoles ──
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Suppress Streamlit / sklearn deprecation noise ──
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

# ── Project root on sys.path ──
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Load production components WITHOUT going through @st.cache_resource.
# We replicate the same initialisation logic but call the constructors
# directly so the test runner has no Streamlit dependency.
# ---------------------------------------------------------------------------
from utils.preprocess import TextPreprocessor
from utils.source_engine import SourceEngine
from utils.clickbait_detector import ClickbaitDetector
from utils.ai_detector import AIContentDetector
from utils.claim_verifier import ClaimVerifier
from src.app import predict_article

# ── ANSI colours ──
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

# ── Pretty separators ──
WIDE  = "=" * 60
THIN  = "-" * 50


# ───────────────────────────────────────────────────────────────
# Helper: load all components once and share across sections
# ───────────────────────────────────────────────────────────────
def load_components():
    """Instantiate every pipeline component directly (no Streamlit cache)."""
    print(f"{DIM}Loading pipeline components...{RESET}")

    preprocessor     = TextPreprocessor()
    source_engine    = SourceEngine()
    clickbait_detect = ClickbaitDetector()
    ai_detect        = AIContentDetector()
    claim_verifier   = ClaimVerifier()

    model_path = ROOT / "model" / "fake_news_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)

    print(f"{GREEN}All components loaded successfully.{RESET}\n")
    return model, preprocessor, clickbait_detect, ai_detect, claim_verifier, source_engine


# ───────────────────────────────────────────────────────────────
# SECTION 1 — Unit & heuristic tests (full pipeline)
# ───────────────────────────────────────────────────────────────
def run_unit_tests(model, preprocessor, clickbait_detect, ai_detect, claim_verifier, source_engine):
    """
    14 hand-crafted cases that exercise every major decision path:
    normal real/fake, extreme fabrication, clickbait, fact-check debunking,
    sensationalism, economy reporting, satire tone, misleading framing,
    source-URL trust, mixed signals, AI-written style, borderline confidence,
    temporal/old-stats, and short-text penalty.
    """
    print(f"\n{CYAN}{BOLD}{WIDE}")
    print("  SECTION 1: UNIT & HEURISTIC TESTS (full predict_article pipeline)")
    print(f"{WIDE}{RESET}\n")

    test_cases = [
        # ── Standard cases ──────────────────────────────────────────────────
        {
            "name": "Normal real news",
            "expected": "REAL",
            "url": None,
            "text": (
                "Central bank raises interest rates by 25 basis points to address "
                "persistent inflation pressures across the economy. Analysts expect "
                "the move to cool consumer spending in the coming quarters."
            ),
        },
        {
            "name": "Normal fake news",
            "expected": "FAKE",
            "url": None,
            "text": "Scientists confirm that dinosaurs still live underground beneath major cities.",
        },

        # ── Extreme fabrication ──────────────────────────────────────────────
        {
            "name": "Extreme fabrication (Immunex Gland)",
            "expected": "FAKE",
            "url": None,
            "text": (
                "BREAKING: Scientists Discover New Organ Inside Human Body That Grants "
                "Immunity to All Diseases. The Immunex Gland produces a unique enzyme "
                "that destroys all pathogens on contact. Activating this organ through "
                "a special diet of rare Himalayan herbs could eliminate the need for all "
                "vaccines and pharmaceutical drugs. Several companies reportedly tried to "
                "suppress the research. Critics note no other team has verified it, and "
                "the World Health Organization declined to comment."
            ),
        },

        # ── Clickbait ────────────────────────────────────────────────────────
        {
            "name": "Scam clickbait",
            "expected": "FAKE",
            "url": None,
            "text": (
                "Secret frequency hidden in FM radio broadcasts can INSTANTLY increase "
                "human IQ by 50 points — doctors HATE this one weird trick!"
            ),
        },

        # ── Fact-check / debunking article ───────────────────────────────────
        {
            "name": "Fact-check debunking article",
            "expected": "REAL",
            "url": None,
            "text": (
                "Fact-checkers say the viral claim about vaccines causing autism is false. "
                "Snopes rated it false. Experts at the WHO and CDC have debunked it "
                "repeatedly. There is no credible peer-reviewed evidence supporting the claim."
            ),
        },

        # ── Sensationalist fake health ────────────────────────────────────────
        {
            "name": "Sensationalist fake health",
            "expected": "FAKE",
            "url": None,
            "text": (
                "Eating ice cream exclusively has been shown to reverse aging completely "
                "and cure diabetes. Big Pharma is hiding this miracle cure from the public."
            ),
        },

        # ── Standard economy reporting ────────────────────────────────────────
        {
            "name": "Standard economy reporting",
            "expected": "REAL",
            "url": None,
            "text": (
                "Economic report highlights gradual recovery in manufacturing activity "
                "and rising exports. The trade deficit narrowed to $68 billion in Q3, "
                "according to the Bureau of Economic Analysis."
            ),
        },

        # ── Satire tone ──────────────────────────────────────────────────────
        {
            "name": "Satirical / parody article",
            "expected": "FAKE",
            "url": None,
            "text": (
                "The Onion reports: Congress Passes Bill Requiring All Americans To Work "
                "Two Full-Time Jobs Just To Pay For Single Full-Time Job's Health Insurance. "
                "President signs legislation after 15-minute debate. Lobbyists celebrate "
                "with champagne on Senate floor as citizens across the nation nod knowingly."
            ),
        },

        # ── Misleading framing ────────────────────────────────────────────────
        {
            "name": "Misleading framing (cherry-picked stat)",
            "expected": "FAKE",
            "url": None,
            "text": (
                "EXCLUSIVE: Crime rises 900% in cities run by liberals! One unnamed study "
                "shows shocking figures the mainstream media refuses to report. "
                "Experts who declined to be named confirmed the trend is accelerating."
            ),
        },

        # ── Source URL trust signal ───────────────────────────────────────────
        {
            "name": "Reuters article via trusted URL",
            "expected": "REAL",
            "url": "https://www.reuters.com/business/finance/federal-reserve-raises-rates-2024",
            "text": (
                "The Federal Reserve raised its benchmark interest rate by a quarter "
                "percentage point on Wednesday, its first increase since January, as "
                "policymakers remain wary of stubborn inflation. The decision was unanimous."
            ),
        },

        # ── Mixed signals (REAL text, low-trust domain) ───────────────────────
        {
            "name": "Credible text but suspicious domain",
            "expected": "FAKE",
            "url": "http://worldnewsdailyreport.com/breaking-story-2024",
            "text": (
                "Scientists at Harvard University announced a major breakthrough in cancer "
                "research. The study, published in Nature, shows a new immunotherapy approach "
                "has a 90% success rate in early trials. The treatment could be available within "
                "five years pending FDA approval."
            ),
        },

        # ── AI-written uniform style ──────────────────────────────────────────
        {
            "name": "AI-generated style misinformation",
            "expected": "FAKE",
            "url": None,
            "text": (
                "Recent studies have demonstrated that consuming activated charcoal every morning "
                "can detoxify the bloodstream, improve cognitive function by up to 40%, and "
                "significantly reduce the risk of chronic illness. Health experts recommend "
                "starting with two tablespoons daily. This natural remedy has been suppressed "
                "by pharmaceutical companies for decades due to its effectiveness."
            ),
        },

        # ── Short-text penalty path ───────────────────────────────────────────
        {
            "name": "Very short text (uncertain penalty path)",
            "expected": "REAL",          # short but benign — should stay uncertain or real
            "url": None,
            "text": "The prime minister signed the new trade agreement today.",
        },

        # ── Temporal / old-stats reuse ────────────────────────────────────────
        {
            "name": "Outdated statistics reused as current",
            "expected": "FAKE",
            "url": None,
            "text": (
                "According to latest 2009 government data, unemployment stands at a record "
                "low of 2.1%. Analysts say the booming economy is unlike anything seen "
                "in decades. Wall Street reacted positively to this week's new figures."
            ),
        },
    ]

    passed = 0
    failed_cases = []

    for tc in test_cases:
        text     = tc["text"]
        expected = tc["expected"]
        url      = tc.get("url")

        res = predict_article(
            text, model, preprocessor,
            clickbait_detect, ai_detect, claim_verifier, source_engine,
            url,
        )

        prediction   = res["prediction"]
        credibility  = res["credibility"]
        ml_score     = res.get("ml_score", 0.5)
        nlp_score    = res.get("nlp_score", 0.5)
        factcheck    = res.get("factcheck_score", 0.5)
        category     = res["category"]
        stance       = res.get("article_stance", "NEUTRAL")
        bert_hit     = res.get("bert_triggered", False)
        redflag_cnt  = res.get("indicators", {}).get("redflag_count", 0)
        evidence_cnt = res.get("evidence_count", 0)

        ok = prediction == expected
        if ok:
            passed += 1
            tag = f"{GREEN}PASS{RESET}"
        else:
            failed_cases.append(tc["name"])
            tag = f"{RED}FAIL{RESET}"

        print(f"[{tag}] {tc['name']}")
        print(f"       Verdict   : {prediction}  (expected {expected})")
        print(
            f"       Scores    : ML={ml_score:.3f}  NLP={nlp_score:.3f}  "
            f"FC={factcheck:.3f}  Cred={credibility:.3f}"
        )
        print(
            f"       Details   : Category={category}  Stance={stance}  "
            f"RedFlags={redflag_cnt}  Evidence={evidence_cnt}  BERT={bert_hit}"
        )
        if url:
            print(f"       URL       : {url}")
        print(THIN)

    pct = passed / len(test_cases) * 100
    colour = GREEN if passed == len(test_cases) else (YELLOW if pct >= 70 else RED)
    print(
        f"\n{BOLD}Unit Test Summary: {colour}{passed}/{len(test_cases)} Passed ({pct:.1f}%){RESET}"
    )
    if failed_cases:
        print(f"{RED}Failed cases:{RESET}")
        for name in failed_cases:
            print(f"  - {name}")
    print()
    return passed == len(test_cases)


# ───────────────────────────────────────────────────────────────
# SECTION 2 — Raw ML benchmark (classifier + preprocessor only)
# ───────────────────────────────────────────────────────────────
def run_ml_benchmark(model, preprocessor, sample_size=10_000):
    """
    Measures raw ML-model accuracy and throughput at scale.
    This intentionally bypasses predict_article — it benchmarks
    only the classifier component (35% of the full scoring weight).
    Results here are NOT the system's end-to-end accuracy.
    """
    print(f"\n{CYAN}{BOLD}{WIDE}")
    print("  SECTION 2: RAW ML BENCHMARK (classifier component only)")
    print(f"  NOTE: This measures 1 of 6 signals — see Section 3 for E2E accuracy")
    print(f"{WIDE}{RESET}\n")

    csv_path = ROOT / "data" / "news.csv"
    if not csv_path.exists():
        print(f"{RED}Error: data/news.csv not found. Skipping ML benchmark.{RESET}\n")
        return

    print(f"Loading corpus: {csv_path}")
    df = pd.read_csv(csv_path)
    sample_size = min(sample_size, len(df))
    df = df.sample(n=sample_size, random_state=42)
    print(f"Sampled {sample_size:,} articles (random_state=42)\n")

    X      = df["text"].fillna("").tolist()
    y_true = df["label"].tolist()

    print("Running preprocessor...")
    t0 = time.perf_counter()
    X_proc = [preprocessor.preprocess_for_model(text) for text in X]

    print("Running classifier predictions...")
    y_pred = model.predict(X_proc)
    elapsed = time.perf_counter() - t0

    throughput  = sample_size / elapsed
    avg_latency = elapsed / sample_size * 1000
    accuracy    = accuracy_score(y_true, y_pred)

    print(f"\n{GREEN}{BOLD}--- ML Benchmark Report ---{RESET}")
    print(f"  Total time    : {elapsed:.2f}s")
    print(f"  Throughput    : {throughput:,.0f} articles/sec")
    print(f"  Avg latency   : {avg_latency:.2f} ms/article")
    print(f"  ML accuracy   : {BOLD}{accuracy*100:.2f}%{RESET}  "
          f"{DIM}(ML component only — not full system){RESET}")
    print("\n  Confusion matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"    {cm}")
    print("\n  Classification report:")
    print(classification_report(y_true, y_pred, digits=4))


# ───────────────────────────────────────────────────────────────
# SECTION 3 — End-to-end stress test (full predict_article)
# ───────────────────────────────────────────────────────────────
def run_e2e_stress_test(
    model, preprocessor, clickbait_detect, ai_detect, claim_verifier, source_engine,
    sample_size=500,
):
    """
    Runs the complete predict_article decision engine on a random sample.
    This is the true system accuracy — all 6 signals, weighted fusion,
    red-flag penalties, stance detection, and verdict thresholds included.

    Default sample_size=500 because the full pipeline is ~10-50x slower
    than the raw classifier (claim verifier, stance detector, etc.).
    Increase to 1000-2000 if you have time to spare.
    """
    print(f"\n{CYAN}{BOLD}{WIDE}")
    print("  SECTION 3: END-TO-END STRESS TEST (full predict_article pipeline)")
    print(f"{WIDE}{RESET}\n")

    csv_path = ROOT / "data" / "news.csv"
    if not csv_path.exists():
        print(f"{RED}Error: data/news.csv not found. Skipping E2E stress test.{RESET}\n")
        return

    print(f"Loading corpus: {csv_path}")
    df = pd.read_csv(csv_path)
    sample_size = min(sample_size, len(df))
    df = df.sample(n=sample_size, random_state=99)
    print(f"Sampled {sample_size:,} articles (random_state=99)\n")

    texts  = df["text"].fillna("").tolist()
    y_true = df["label"].tolist()

    y_pred       = []
    categories   = []
    credibilities = []
    errors       = 0

    print(f"Running full pipeline on {sample_size:,} articles...")
    print(f"{DIM}(This will take longer than Section 2 — each article runs all 6 signals){RESET}\n")

    t0 = time.perf_counter()
    for i, text in enumerate(texts, 1):
        try:
            res = predict_article(
                text, model, preprocessor,
                clickbait_detect, ai_detect, claim_verifier, source_engine,
                None,  # no URL available in the CSV dataset
            )
            y_pred.append(res["prediction"])
            categories.append(res["category"])
            credibilities.append(res["credibility"])
        except Exception as exc:
            # Don't let one bad article crash the whole run
            y_pred.append("REAL")   # neutral fallback so metrics stay defined
            categories.append("Error")
            credibilities.append(0.5)
            errors += 1

        # Progress ticker every 50 articles
        if i % 50 == 0 or i == sample_size:
            elapsed_so_far = time.perf_counter() - t0
            rate = i / elapsed_so_far
            eta  = (sample_size - i) / rate if rate > 0 else 0
            print(f"  {i:>5}/{sample_size}  |  {rate:5.1f} art/s  |  ETA {eta:5.1f}s", end="\r")

    elapsed = time.perf_counter() - t0
    print()  # newline after \r ticker

    throughput  = sample_size / elapsed
    avg_latency = elapsed / sample_size * 1000
    accuracy    = accuracy_score(y_true, y_pred)

    # Category distribution
    from collections import Counter
    cat_dist = Counter(categories)

    # Average credibility score per true label
    real_creds = [c for c, l in zip(credibilities, y_true) if l == "REAL"]
    fake_creds = [c for c, l in zip(credibilities, y_true) if l == "FAKE"]
    avg_real_cred = sum(real_creds) / len(real_creds) if real_creds else 0
    avg_fake_cred = sum(fake_creds) / len(fake_creds) if fake_creds else 0

    print(f"\n{GREEN}{BOLD}--- E2E Stress Test Report ---{RESET}")
    print(f"  Sample size   : {sample_size:,} articles")
    print(f"  Total time    : {elapsed:.2f}s")
    print(f"  Throughput    : {throughput:.1f} articles/sec")
    print(f"  Avg latency   : {avg_latency:.1f} ms/article")
    print(f"  Pipeline err  : {errors} articles raised exceptions")
    print(f"\n  {BOLD}E2E system accuracy: {accuracy*100:.2f}%{RESET}")
    print()

    print("  Confusion matrix (REAL articles = positive class):")
    cm = confusion_matrix(y_true, y_pred, labels=["REAL", "FAKE"])
    print(f"             Pred REAL  Pred FAKE")
    print(f"  True REAL  {cm[0][0]:>9}  {cm[0][1]:>9}")
    print(f"  True FAKE  {cm[1][0]:>9}  {cm[1][1]:>9}")

    print("\n  Classification report:")
    print(classification_report(y_true, y_pred, digits=4))

    print("  5-level verdict distribution:")
    for cat, cnt in sorted(cat_dist.items(), key=lambda x: -x[1]):
        pct = cnt / sample_size * 100
        print(f"    {cat:<30} {cnt:>5}  ({pct:.1f}%)")

    print(f"\n  Avg credibility score:")
    print(f"    True REAL articles : {avg_real_cred:.3f}")
    print(f"    True FAKE articles : {avg_fake_cred:.3f}")
    print(f"    Delta              : {avg_real_cred - avg_fake_cred:+.3f}  "
          f"{DIM}(higher = better separation){RESET}")
    print()


# ───────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────
def main():
    print(f"\n{CYAN}{BOLD}")
    print("+------------------------------------------------------+")
    print("|       TruthShield Unified Test Suite v2              |")
    print("|  Section 1: Unit tests    (14 cases, full pipeline)  |")
    print("|  Section 2: ML benchmark  (10,000 articles, fast)    |")
    print("|  Section 3: E2E stress    (500 articles, full pipe)  |")
    print(f"+------------------------------------------------------+{RESET}\n")

    try:
        model, preprocessor, clickbait_detect, ai_detect, claim_verifier, source_engine = (
            load_components()
        )
    except Exception as exc:
        print(f"{RED}{BOLD}Fatal: could not load pipeline components.{RESET}")
        print(f"{RED}{exc}{RESET}")
        sys.exit(1)

    # ── Section 1 ──
    run_unit_tests(
        model, preprocessor, clickbait_detect, ai_detect, claim_verifier, source_engine
    )

    # ── Section 2 ──
    run_ml_benchmark(model, preprocessor, sample_size=10_000)

    # ── Section 3 ──
    # Increase sample_size here if you want more statistical power.
    # 500 completes in ~2-5 minutes depending on your machine.
    run_e2e_stress_test(
        model, preprocessor, clickbait_detect, ai_detect, claim_verifier, source_engine,
        sample_size=500,
    )


if __name__ == "__main__":
    main()
