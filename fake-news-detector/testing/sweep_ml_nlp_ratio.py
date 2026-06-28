"""
TruthShield Diagnostic — Weight Ratio Sweep

The previous diagnostic showed: in the no-source/no-factcheck case, ml_score
is already correctly leaning FAKE on missed articles (avg 0.335) but nlp_score
fights it hard (avg 0.744), and nlp_score's weight is large enough to win.

This script does NOT change app.py. It monkey-patches the w_ml/w_nlp split at
runtime for several candidate ratios, re-runs the SAME 500-article sample
(random_state=99) AND the 14 unit-test cases for each candidate, and reports:
  - FAKE recall / REAL recall / Uncertain rate for each candidate
  - whether any of the 14 unit tests flip from PASS to FAIL

This lets us pick a ratio backed by full-pipeline numbers instead of hand
arithmetic, and see the unit-test regression risk before touching app.py.

Usage (from project root, same venv as test_suite.py):
    python testing\\sweep_ml_nlp_ratio.py
"""

import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import joblib
from sklearn.metrics import recall_score

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.preprocess import TextPreprocessor
from utils.source_engine import SourceEngine
from utils.clickbait_detector import ClickbaitDetector
from utils.ai_detector import AIContentDetector
from utils.claim_verifier import ClaimVerifier
import src.app as app_module

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# ── The 14 unit test cases, copied from test_suite.py so we can check
#    regression risk for each candidate ratio without modifying app.py ──
UNIT_TESTS = [
    {"name": "Normal real news", "expected": "REAL", "url": None,
     "text": ("Central bank raises interest rates by 25 basis points to address "
               "persistent inflation pressures across the economy. Analysts expect "
               "the move to cool consumer spending in the coming quarters.")},
    {"name": "Normal fake news", "expected": "FAKE", "url": None,
     "text": "Scientists confirm that dinosaurs still live underground beneath major cities."},
    {"name": "Extreme fabrication (Immunex Gland)", "expected": "FAKE", "url": None,
     "text": ("BREAKING: Scientists Discover New Organ Inside Human Body That Grants "
               "Immunity to All Diseases. The Immunex Gland produces a unique enzyme "
               "that destroys all pathogens on contact. Activating this organ through "
               "a special diet of rare Himalayan herbs could eliminate the need for all "
               "vaccines and pharmaceutical drugs. Several companies reportedly tried to "
               "suppress the research. Critics note no other team has verified it, and "
               "the World Health Organization declined to comment.")},
    {"name": "Scam clickbait", "expected": "FAKE", "url": None,
     "text": ("Secret frequency hidden in FM radio broadcasts can INSTANTLY increase "
               "human IQ by 50 points — doctors HATE this one weird trick!")},
    {"name": "Fact-check debunking article", "expected": "REAL", "url": None,
     "text": ("Fact-checkers say the viral claim about vaccines causing autism is false. "
               "Snopes rated it false. Experts at the WHO and CDC have debunked it "
               "repeatedly. There is no credible peer-reviewed evidence supporting the claim.")},
    {"name": "Sensationalist fake health", "expected": "FAKE", "url": None,
     "text": ("Eating ice cream exclusively has been shown to reverse aging completely "
               "and cure diabetes. Big Pharma is hiding this miracle cure from the public.")},
    {"name": "Standard economy reporting", "expected": "REAL", "url": None,
     "text": ("Economic report highlights gradual recovery in manufacturing activity "
               "and rising exports. The trade deficit narrowed to $68 billion in Q3, "
               "according to the Bureau of Economic Analysis.")},
    {"name": "Satirical / parody article", "expected": "FAKE", "url": None,
     "text": ("The Onion reports: Congress Passes Bill Requiring All Americans To Work "
               "Two Full-Time Jobs Just To Pay For Single Full-Time Job's Health Insurance. "
               "President signs legislation after 15-minute debate. Lobbyists celebrate "
               "with champagne on Senate floor as citizens across the nation nod knowingly.")},
    {"name": "Misleading framing (cherry-picked stat)", "expected": "FAKE", "url": None,
     "text": ("EXCLUSIVE: Crime rises 900% in cities run by liberals! One unnamed study "
               "shows shocking figures the mainstream media refuses to report. "
               "Experts who declined to be named confirmed the trend is accelerating.")},
    {"name": "Reuters article via trusted URL", "expected": "REAL",
     "url": "https://www.reuters.com/business/finance/federal-reserve-raises-rates-2024",
     "text": ("The Federal Reserve raised its benchmark interest rate by a quarter "
               "percentage point on Wednesday, its first increase since January, as "
               "policymakers remain wary of stubborn inflation. The decision was unanimous.")},
    {"name": "Credible text but suspicious domain", "expected": "FAKE",
     "url": "http://worldnewsdailyreport.com/breaking-story-2024",
     "text": ("Scientists at Harvard University announced a major breakthrough in cancer "
               "research. The study, published in Nature, shows a new immunotherapy approach "
               "has a 90% success rate in early trials. The treatment could be available within "
               "five years pending FDA approval.")},
    {"name": "AI-generated style misinformation", "expected": "FAKE", "url": None,
     "text": ("Recent studies have demonstrated that consuming activated charcoal every morning "
               "can detoxify the bloodstream, improve cognitive function by up to 40%, and "
               "significantly reduce the risk of chronic illness. Health experts recommend "
               "starting with two tablespoons daily. This natural remedy has been suppressed "
               "by pharmaceutical companies for decades due to its effectiveness.")},
    {"name": "Very short text (uncertain penalty path)", "expected": "REAL", "url": None,
     "text": "The prime minister signed the new trade agreement today."},
    {"name": "Outdated statistics reused as current", "expected": "FAKE", "url": None,
     "text": ("According to latest 2009 government data, unemployment stands at a record "
               "low of 2.1%. Analysts say the booming economy is unlike anything seen "
               "in decades. Wall Street reacted positively to this week's new figures.")},
]


def patch_ml_nlp_pool(ratio):
    """
    Monkey-patch the no-source/no-factcheck weight split inside predict_article
    by wrapping it: we can't easily edit the function body at runtime, so instead
    we patch module-level base weights w_ml/w_nlp are computed from, IF app.py
    exposes them as constants. If not exposed, fall back to env var + re-import.

    Simpler and more robust approach: monkeypatch is skipped here; instead this
    script calls predict_article normally and POST-HOC recomputes what credibility
    WOULD have been under a different ratio, using the ml_score/nlp_score/etc
    that predict_article already returns. This avoids touching app.py entirely
    and avoids any risk of corrupting your real pipeline state.
    """
    pass  # see recompute_credibility() below — the actual logic lives there


def recompute_credibility(res, ratio):
    """
    Given a predict_article() result dict, recompute what `credibility` and the
    final verdict would have been if w_ml/w_nlp (in the no-source/no-factcheck
    case ONLY) used the given ratio instead of the current 0.585/0.285 split,
    holding the ml_nlp pool size (0.87) and all other weights fixed.

    This is post-hoc and only valid for has_source_signal=False AND
    has_factcheck_signal=False, which is true for ~all 500 CSV articles
    (no URL is passed in the E2E test) and the relevant unit tests above.
    """
    ml_score = res["ml_score"]
    nlp_score = res["nlp_score"]
    ai_score = res["ai_score"]
    clickbait_score = res["clickbait_score"]
    satire_score = res.get("satire_score", 0.0)
    indicators = res.get("indicators", {})
    redflag_count = indicators.get("redflag_count", 0)
    temporal_penalty = indicators.get("temporal_penalty", 0.0)
    factcheck_score = res.get("factcheck_score", 0.5)
    source_trust = res.get("source_trust", 50.0)
    stance = res.get("article_stance", "NEUTRAL")
    stance_confidence = res.get("stance_confidence", 0.5)

    # Only recompute for the no-signal case; otherwise just return the original
    has_source_signal = source_trust != 50.0  # approximation; good enough for CSV (no URL)
    has_factcheck_signal = res.get("evidence_count", 0) > 0
    if has_source_signal or has_factcheck_signal:
        return res["credibility"], res["prediction"]

    ml_nlp_pool = 0.585 + 0.285  # = 0.87, fixed
    w_nlp = ml_nlp_pool / (ratio + 1)
    w_ml = ml_nlp_pool - w_nlp
    w_clickbait = 0.025
    w_ai = 0.025
    w_satire = 0.08

    credibility = (
        ml_score * w_ml +
        nlp_score * w_nlp +
        (1.0 - clickbait_score) * w_clickbait +
        (1.0 - ai_score) * w_ai +
        (1.0 - satire_score) * w_satire
    )

    # ── Stance-aware adjustment (must mirror app.py exactly) ──
    factchecker_refs = res.get("stance", {}).get("factchecker_references", [])
    cred_signal = indicators.get("credibility_score", 0.0)
    if stance == "REFUTES" and stance_confidence >= 0.2:
        if res.get("prediction") == "FAKE":
            ml_boost = (1.0 - ml_score) * stance_confidence * 0.35
            credibility += ml_boost
        refutation_boost = stance_confidence * 0.15
        credibility += refutation_boost
        if factchecker_refs:
            credibility += min(len(factchecker_refs) * 0.05, 0.15)
        cred_indicator_boost = cred_signal * 0.1
        credibility += cred_indicator_boost
    elif stance == "SUPPORTS" and stance_confidence >= 0.3:
        credibility -= stance_confidence * 0.1

    if redflag_count > 0 and stance != "REFUTES":
        redflag_penalty = min(redflag_count * 0.10, 0.55)
        credibility -= redflag_penalty
        if redflag_count >= 5:
            credibility = min(credibility, 0.18)
        elif redflag_count >= 3:
            credibility = min(credibility, 0.42)

    credibility = float(max(0.0, min(1.0, credibility)))
    credibility -= temporal_penalty
    credibility = float(max(0.0, min(1.0, credibility)))

    if nlp_score < 0.4:
        threshold = 0.50
    elif redflag_count >= 2:
        threshold = 0.47
    else:
        threshold = 0.55

    prediction = "REAL" if credibility >= threshold else "FAKE"

    if res.get("is_satire") and satire_score > 0.65:
        prediction = "FAKE"
        credibility = min(credibility, 0.35)

    return credibility, prediction


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
    df = df.sample(n=sample_size, random_state=99)
    texts = df["text"].fillna("").tolist()
    y_true = df["label"].tolist()

    print(f"{GREEN}Running base pipeline once on {sample_size} articles to collect raw signals...{RESET}\n")

    base_results = []
    for text in texts:
        try:
            res = app_module.predict_article(
                text, model, preprocessor,
                clickbait_detect, ai_detect, claim_verifier, source_engine,
                None,
            )
            base_results.append(res)
        except Exception:
            base_results.append(None)

    print(f"{GREEN}Running 14 unit tests once to collect raw signals...{RESET}\n")
    unit_base_results = []
    for tc in UNIT_TESTS:
        try:
            res = app_module.predict_article(
                tc["text"], model, preprocessor,
                clickbait_detect, ai_detect, claim_verifier, source_engine,
                tc.get("url"),
            )
            unit_base_results.append(res)
        except Exception:
            unit_base_results.append(None)

    candidate_ratios = [2.05, 2.5, 3.0, 3.5, 4.0, 4.4, 5.0, 6.0]

    print(f"{BOLD}{'='*90}{RESET}")
    print(f"{BOLD}  Ratio sweep — 500-article E2E corpus{RESET}")
    print(f"{'='*90}{RESET}")
    print(f"{'ratio':>6} {'w_ml':>7} {'w_nlp':>7} {'FAKE recall':>12} {'REAL recall':>12} {'Uncertain%':>11}")

    for ratio in candidate_ratios:
        y_pred = []
        creds = []
        for text, label, res in zip(texts, y_true, base_results):
            if res is None:
                y_pred.append("REAL")
                creds.append(0.5)
                continue
            cred, pred = recompute_credibility(res, ratio)
            y_pred.append(pred)
            creds.append(cred)

        fake_recall = recall_score(y_true, y_pred, pos_label="FAKE", labels=["FAKE", "REAL"])
        real_recall = recall_score(y_true, y_pred, pos_label="REAL", labels=["FAKE", "REAL"])
        uncertain_pct = sum(1 for c in creds if 0.45 <= c < 0.65) / len(creds) * 100

        w_nlp = 0.87 / (ratio + 1)
        w_ml = 0.87 - w_nlp
        print(f"{ratio:>6.2f} {w_ml:>7.3f} {w_nlp:>7.3f} {fake_recall*100:>11.1f}% {real_recall*100:>11.1f}% {uncertain_pct:>10.1f}%")

    print()
    print(f"{BOLD}{'='*90}{RESET}")
    print(f"{BOLD}  Ratio sweep — 14 unit tests (showing only ratios where a test flips){RESET}")
    print(f"{'='*90}{RESET}\n")

    baseline_ratio = 2.0526315789473686  # current app.py ratio (0.585/0.285)
    baseline_preds = []
    for res in unit_base_results:
        if res is None:
            baseline_preds.append("REAL")
            continue
        _, pred = recompute_credibility(res, baseline_ratio)
        baseline_preds.append(pred)

    for ratio in candidate_ratios:
        flips = []
        for tc, res, base_pred in zip(UNIT_TESTS, unit_base_results, baseline_preds):
            if res is None:
                continue
            cred, pred = recompute_credibility(res, ratio)
            if pred != base_pred:
                status_before = "PASS" if base_pred == tc["expected"] else "FAIL"
                status_after = "PASS" if pred == tc["expected"] else "FAIL"
                flips.append((tc["name"], base_pred, pred, status_before, status_after, cred))

        if flips:
            print(f"{YELLOW}{BOLD}ratio={ratio:.2f}:{RESET}")
            for name, before, after, sb, sa, cred in flips:
                arrow = f"{before}->{after}"
                tag = f"{GREEN}{sb}->{sa}{RESET}" if sa == "PASS" else f"{RED}{sb}->{sa}{RESET}"
                print(f"    {name:<40} {arrow:<14} cred={cred:.3f}  [{tag}]")
        else:
            print(f"{DIM}ratio={ratio:.2f}: no unit test changes{RESET}")
    print()


if __name__ == "__main__":
    main()
"""
TruthShield Diagnostic — Weight Ratio Sweep

The previous diagnostic showed: in the no-source/no-factcheck case, ml_score
is already correctly leaning FAKE on missed articles (avg 0.335) but nlp_score
fights it hard (avg 0.744), and nlp_score's weight is large enough to win.

This script does NOT change app.py. It monkey-patches the w_ml/w_nlp split at
runtime for several candidate ratios, re-runs the SAME 500-article sample
(random_state=99) AND the 14 unit-test cases for each candidate, and reports:
  - FAKE recall / REAL recall / Uncertain rate for each candidate
  - whether any of the 14 unit tests flip from PASS to FAIL

This lets us pick a ratio backed by full-pipeline numbers instead of hand
arithmetic, and see the unit-test regression risk before touching app.py.

Usage (from project root, same venv as test_suite.py):
    python testing\\sweep_ml_nlp_ratio.py
"""

import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import joblib
from sklearn.metrics import recall_score

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.preprocess import TextPreprocessor
from utils.source_engine import SourceEngine
from utils.clickbait_detector import ClickbaitDetector
from utils.ai_detector import AIContentDetector
from utils.claim_verifier import ClaimVerifier
import src.app as app_module

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# ── The 14 unit test cases, copied from test_suite.py so we can check
#    regression risk for each candidate ratio without modifying app.py ──
UNIT_TESTS = [
    {"name": "Normal real news", "expected": "REAL", "url": None,
     "text": ("Central bank raises interest rates by 25 basis points to address "
               "persistent inflation pressures across the economy. Analysts expect "
               "the move to cool consumer spending in the coming quarters.")},
    {"name": "Normal fake news", "expected": "FAKE", "url": None,
     "text": "Scientists confirm that dinosaurs still live underground beneath major cities."},
    {"name": "Extreme fabrication (Immunex Gland)", "expected": "FAKE", "url": None,
     "text": ("BREAKING: Scientists Discover New Organ Inside Human Body That Grants "
               "Immunity to All Diseases. The Immunex Gland produces a unique enzyme "
               "that destroys all pathogens on contact. Activating this organ through "
               "a special diet of rare Himalayan herbs could eliminate the need for all "
               "vaccines and pharmaceutical drugs. Several companies reportedly tried to "
               "suppress the research. Critics note no other team has verified it, and "
               "the World Health Organization declined to comment.")},
    {"name": "Scam clickbait", "expected": "FAKE", "url": None,
     "text": ("Secret frequency hidden in FM radio broadcasts can INSTANTLY increase "
               "human IQ by 50 points — doctors HATE this one weird trick!")},
    {"name": "Fact-check debunking article", "expected": "REAL", "url": None,
     "text": ("Fact-checkers say the viral claim about vaccines causing autism is false. "
               "Snopes rated it false. Experts at the WHO and CDC have debunked it "
               "repeatedly. There is no credible peer-reviewed evidence supporting the claim.")},
    {"name": "Sensationalist fake health", "expected": "FAKE", "url": None,
     "text": ("Eating ice cream exclusively has been shown to reverse aging completely "
               "and cure diabetes. Big Pharma is hiding this miracle cure from the public.")},
    {"name": "Standard economy reporting", "expected": "REAL", "url": None,
     "text": ("Economic report highlights gradual recovery in manufacturing activity "
               "and rising exports. The trade deficit narrowed to $68 billion in Q3, "
               "according to the Bureau of Economic Analysis.")},
    {"name": "Satirical / parody article", "expected": "FAKE", "url": None,
     "text": ("The Onion reports: Congress Passes Bill Requiring All Americans To Work "
               "Two Full-Time Jobs Just To Pay For Single Full-Time Job's Health Insurance. "
               "President signs legislation after 15-minute debate. Lobbyists celebrate "
               "with champagne on Senate floor as citizens across the nation nod knowingly.")},
    {"name": "Misleading framing (cherry-picked stat)", "expected": "FAKE", "url": None,
     "text": ("EXCLUSIVE: Crime rises 900% in cities run by liberals! One unnamed study "
               "shows shocking figures the mainstream media refuses to report. "
               "Experts who declined to be named confirmed the trend is accelerating.")},
    {"name": "Reuters article via trusted URL", "expected": "REAL",
     "url": "https://www.reuters.com/business/finance/federal-reserve-raises-rates-2024",
     "text": ("The Federal Reserve raised its benchmark interest rate by a quarter "
               "percentage point on Wednesday, its first increase since January, as "
               "policymakers remain wary of stubborn inflation. The decision was unanimous.")},
    {"name": "Credible text but suspicious domain", "expected": "FAKE",
     "url": "http://worldnewsdailyreport.com/breaking-story-2024",
     "text": ("Scientists at Harvard University announced a major breakthrough in cancer "
               "research. The study, published in Nature, shows a new immunotherapy approach "
               "has a 90% success rate in early trials. The treatment could be available within "
               "five years pending FDA approval.")},
    {"name": "AI-generated style misinformation", "expected": "FAKE", "url": None,
     "text": ("Recent studies have demonstrated that consuming activated charcoal every morning "
               "can detoxify the bloodstream, improve cognitive function by up to 40%, and "
               "significantly reduce the risk of chronic illness. Health experts recommend "
               "starting with two tablespoons daily. This natural remedy has been suppressed "
               "by pharmaceutical companies for decades due to its effectiveness.")},
    {"name": "Very short text (uncertain penalty path)", "expected": "REAL", "url": None,
     "text": "The prime minister signed the new trade agreement today."},
    {"name": "Outdated statistics reused as current", "expected": "FAKE", "url": None,
     "text": ("According to latest 2009 government data, unemployment stands at a record "
               "low of 2.1%. Analysts say the booming economy is unlike anything seen "
               "in decades. Wall Street reacted positively to this week's new figures.")},
]


def patch_ml_nlp_pool(ratio):
    """
    Monkey-patch the no-source/no-factcheck weight split inside predict_article
    by wrapping it: we can't easily edit the function body at runtime, so instead
    we patch module-level base weights w_ml/w_nlp are computed from, IF app.py
    exposes them as constants. If not exposed, fall back to env var + re-import.

    Simpler and more robust approach: monkeypatch is skipped here; instead this
    script calls predict_article normally and POST-HOC recomputes what credibility
    WOULD have been under a different ratio, using the ml_score/nlp_score/etc
    that predict_article already returns. This avoids touching app.py entirely
    and avoids any risk of corrupting your real pipeline state.
    """
    pass  # see recompute_credibility() below — the actual logic lives there


def recompute_credibility(res, ratio):
    """
    Given a predict_article() result dict, recompute what `credibility` and the
    final verdict would have been if w_ml/w_nlp (in the no-source/no-factcheck
    case ONLY) used the given ratio instead of the current 0.585/0.285 split,
    holding the ml_nlp pool size (0.87) and all other weights fixed.

    This is post-hoc and only valid for has_source_signal=False AND
    has_factcheck_signal=False, which is true for ~all 500 CSV articles
    (no URL is passed in the E2E test) and the relevant unit tests above.
    """
    ml_score = res["ml_score"]
    nlp_score = res["nlp_score"]
    ai_score = res["ai_score"]
    clickbait_score = res["clickbait_score"]
    satire_score = res.get("satire_score", 0.0)
    indicators = res.get("indicators", {})
    redflag_count = indicators.get("redflag_count", 0)
    temporal_penalty = indicators.get("temporal_penalty", 0.0)
    factcheck_score = res.get("factcheck_score", 0.5)
    source_trust = res.get("source_trust", 50.0)
    stance = res.get("article_stance", "NEUTRAL")
    stance_confidence = res.get("stance_confidence", 0.5)

    # Only recompute for the no-signal case; otherwise just return the original
    has_source_signal = source_trust != 50.0  # approximation; good enough for CSV (no URL)
    has_factcheck_signal = res.get("evidence_count", 0) > 0
    if has_source_signal or has_factcheck_signal:
        return res["credibility"], res["prediction"]

    ml_nlp_pool = 0.585 + 0.285  # = 0.87, fixed
    w_nlp = ml_nlp_pool / (ratio + 1)
    w_ml = ml_nlp_pool - w_nlp
    w_clickbait = 0.025
    w_ai = 0.025
    w_satire = 0.08

    credibility = (
        ml_score * w_ml +
        nlp_score * w_nlp +
        (1.0 - clickbait_score) * w_clickbait +
        (1.0 - ai_score) * w_ai +
        (1.0 - satire_score) * w_satire
    )

    # ── Stance-aware adjustment (must mirror app.py exactly) ──
    factchecker_refs = res.get("stance", {}).get("factchecker_references", [])
    cred_signal = indicators.get("credibility_score", 0.0)
    if stance == "REFUTES" and stance_confidence >= 0.2:
        if res.get("prediction") == "FAKE":
            ml_boost = (1.0 - ml_score) * stance_confidence * 0.35
            credibility += ml_boost
        refutation_boost = stance_confidence * 0.15
        credibility += refutation_boost
        if factchecker_refs:
            credibility += min(len(factchecker_refs) * 0.05, 0.15)
        cred_indicator_boost = cred_signal * 0.1
        credibility += cred_indicator_boost
    elif stance == "SUPPORTS" and stance_confidence >= 0.3:
        credibility -= stance_confidence * 0.1

    if redflag_count > 0 and stance != "REFUTES":
        redflag_penalty = min(redflag_count * 0.10, 0.55)
        credibility -= redflag_penalty
        if redflag_count >= 5:
            credibility = min(credibility, 0.18)
        elif redflag_count >= 3:
            credibility = min(credibility, 0.42)

    credibility = float(max(0.0, min(1.0, credibility)))
    credibility -= temporal_penalty
    credibility = float(max(0.0, min(1.0, credibility)))

    if nlp_score < 0.4:
        threshold = 0.50
    elif redflag_count >= 2:
        threshold = 0.47
    else:
        threshold = 0.55

    prediction = "REAL" if credibility >= threshold else "FAKE"

    if res.get("is_satire") and satire_score > 0.65:
        prediction = "FAKE"
        credibility = min(credibility, 0.35)

    return credibility, prediction


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
    df = df.sample(n=sample_size, random_state=99)
    texts = df["text"].fillna("").tolist()
    y_true = df["label"].tolist()

    print(f"{GREEN}Running base pipeline once on {sample_size} articles to collect raw signals...{RESET}\n")

    base_results = []
    for text in texts:
        try:
            res = app_module.predict_article(
                text, model, preprocessor,
                clickbait_detect, ai_detect, claim_verifier, source_engine,
                None,
            )
            base_results.append(res)
        except Exception:
            base_results.append(None)

    print(f"{GREEN}Running 14 unit tests once to collect raw signals...{RESET}\n")
    unit_base_results = []
    for tc in UNIT_TESTS:
        try:
            res = app_module.predict_article(
                tc["text"], model, preprocessor,
                clickbait_detect, ai_detect, claim_verifier, source_engine,
                tc.get("url"),
            )
            unit_base_results.append(res)
        except Exception:
            unit_base_results.append(None)

    candidate_ratios = [2.05, 2.5, 3.0, 3.5, 4.0, 4.4, 5.0, 6.0]

    print(f"{BOLD}{'='*90}{RESET}")
    print(f"{BOLD}  Ratio sweep — 500-article E2E corpus{RESET}")
    print(f"{'='*90}{RESET}")
    print(f"{'ratio':>6} {'w_ml':>7} {'w_nlp':>7} {'FAKE recall':>12} {'REAL recall':>12} {'Uncertain%':>11}")

    for ratio in candidate_ratios:
        y_pred = []
        creds = []
        for text, label, res in zip(texts, y_true, base_results):
            if res is None:
                y_pred.append("REAL")
                creds.append(0.5)
                continue
            cred, pred = recompute_credibility(res, ratio)
            y_pred.append(pred)
            creds.append(cred)

        fake_recall = recall_score(y_true, y_pred, pos_label="FAKE", labels=["FAKE", "REAL"])
        real_recall = recall_score(y_true, y_pred, pos_label="REAL", labels=["FAKE", "REAL"])
        uncertain_pct = sum(1 for c in creds if 0.45 <= c < 0.65) / len(creds) * 100

        w_nlp = 0.87 / (ratio + 1)
        w_ml = 0.87 - w_nlp
        print(f"{ratio:>6.2f} {w_ml:>7.3f} {w_nlp:>7.3f} {fake_recall*100:>11.1f}% {real_recall*100:>11.1f}% {uncertain_pct:>10.1f}%")

    print()
    print(f"{BOLD}{'='*90}{RESET}")
    print(f"{BOLD}  Ratio sweep — 14 unit tests (showing only ratios where a test flips){RESET}")
    print(f"{'='*90}{RESET}\n")

    baseline_ratio = 2.0526315789473686  # current app.py ratio (0.585/0.285)
    baseline_preds = []
    for res in unit_base_results:
        if res is None:
            baseline_preds.append("REAL")
            continue
        _, pred = recompute_credibility(res, baseline_ratio)
        baseline_preds.append(pred)

    for ratio in candidate_ratios:
        flips = []
        for tc, res, base_pred in zip(UNIT_TESTS, unit_base_results, baseline_preds):
            if res is None:
                continue
            cred, pred = recompute_credibility(res, ratio)
            if pred != base_pred:
                status_before = "PASS" if base_pred == tc["expected"] else "FAIL"
                status_after = "PASS" if pred == tc["expected"] else "FAIL"
                flips.append((tc["name"], base_pred, pred, status_before, status_after, cred))

        if flips:
            print(f"{YELLOW}{BOLD}ratio={ratio:.2f}:{RESET}")
            for name, before, after, sb, sa, cred in flips:
                arrow = f"{before}->{after}"
                tag = f"{GREEN}{sb}->{sa}{RESET}" if sa == "PASS" else f"{RED}{sb}->{sa}{RESET}"
                print(f"    {name:<40} {arrow:<14} cred={cred:.3f}  [{tag}]")
        else:
            print(f"{DIM}ratio={ratio:.2f}: no unit test changes{RESET}")
    print()


if __name__ == "__main__":
    main()