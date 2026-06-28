"""
API Server for TruthShield Fake News Detector.

Provides REST endpoints for predictions, advanced analyses, user feedback,
and historical search. Integrates with the database, clickbait detector,
AI content detector, claim verifier, multilingual translator, and source trust engine.
"""

import os
import sys
import io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import sqlite3
import joblib
import json
import re
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Imports from utils
from utils.preprocess import TextPreprocessor
from utils.clickbait_detector import ClickbaitDetector
from utils.ai_detector import AIContentDetector
from utils.claim_verifier import ClaimVerifier
from utils.multilingual import MultilingualProcessor
from utils.source_engine import SourceEngine
from utils.stance_detector import StanceDetector

app = FastAPI(
    title="TruthShield Production API",
    description="Backend API for TruthShield Credibility Checker",
    version="2.0.0"
)

# CORS middleware for browser extensions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
preprocessor = None
clickbait_detector = None
ai_detector = None
claim_verifier = None
multilingual_processor = None
source_engine = None
stance_detector = None

model_path = os.path.join(PROJECT_ROOT, "model", "fake_news_model.pkl")
db_path = os.path.join(PROJECT_ROOT, "data", "truthshield.db")
last_loaded_mtime = 0.0

# ------------------------------------------------------------------
# Request & Response Schemas
# ------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=20, description="Article text or content to analyze")
    url: Optional[str] = Field(None, description="Optional article source URL")

class PredictResponse(BaseModel):
    prediction: str
    article_hash: str
    confidence: float
    credibility: float
    reliability: float
    summary: str
    is_fake: bool

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=20, description="Article text to analyze")
    url: Optional[str] = Field(None, description="Optional article source URL")

class VerificationResultItem(BaseModel):
    claim: str
    rating: str
    url: Optional[str] = None
    source: Optional[str] = None

class AnalyzeResponse(BaseModel):
    prediction: str
    article_hash: str
    confidence: float
    credibility: float
    reliability: float
    category: str
    clickbait_score: float
    ai_score: float
    source_trust: float
    verification_results: List[Dict]
    summary: str
    language: str
    was_translated: bool

class FeedbackRequest(BaseModel):
    user_email: Optional[str] = "anonymous@truthshield.ai"
    text: str
    model_prediction: str
    user_verdict: str
    rating: Optional[int] = 5
    notes: Optional[str] = ""

# ------------------------------------------------------------------
# Database Functions & Migrations
# ------------------------------------------------------------------

def get_db_connection():
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create database tables and handle migrations if necessary."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            last_login DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            title TEXT,
            text TEXT,
            prediction TEXT,
            confidence REAL,
            credibility REAL,
            reliability REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            category TEXT,
            clickbait_score REAL,
            ai_score REAL,
            source_trust REAL,
            verification_status TEXT,
            FOREIGN KEY(user_email) REFERENCES users(email)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            text TEXT,
            model_prediction TEXT,
            user_verdict TEXT,
            rating INTEGER,
            notes TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_email) REFERENCES users(email)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS monitoring_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            claims_fetched INTEGER,
            claims_new INTEGER,
            source TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS source_reputation (
            domain TEXT PRIMARY KEY,
            trust_score REAL,
            bias TEXT,
            category TEXT,
            description TEXT,
            fact_check_history TEXT,
            accuracy_rate REAL DEFAULT 1.0,
            frequency INTEGER DEFAULT 0
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS claims_kb (
            claim_hash TEXT PRIMARY KEY,
            claim_text TEXT,
            verdict TEXT,
            source TEXT,
            details TEXT,
            priority INTEGER DEFAULT 1,
            last_verified DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_hash TEXT,
            verdict TEXT,
            confidence REAL,
            credibility REAL,
            reliability REAL,
            source_score REAL,
            factcheck_score REAL,
            clickbait_score REAL,
            ai_score REAL,
            features_contribution TEXT,
            monthly_accuracy REAL DEFAULT 0.90,
            source_distribution TEXT,
            prediction_distribution TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS misinformation_patterns (
            pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
            theme TEXT,
            regex_pattern TEXT,
            risk_multiplier REAL
        )
    """)
    
    # Phase 8: Continuous Learning — misclassifications tracking table
    cursor.execute("""
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
    
    # Run migration checks: Ensure newer columns are added to existing history table
    try:
        cursor.execute("PRAGMA table_info(history)")
        columns = [row[1] for row in cursor.fetchall()]
        
        new_cols = {
            "category": "TEXT",
            "clickbait_score": "REAL",
            "ai_score": "REAL",
            "source_trust": "REAL",
            "verification_status": "TEXT",
            "reliability": "REAL"
        }
        
        for col, col_type in new_cols.items():
            if col not in columns:
                cursor.execute(f"ALTER TABLE history ADD COLUMN {col} {col_type}")
                print(f"🔧 Database migration: added '{col}' to history table.")
    except Exception as e:
        print(f"⚠️ Database migration error: {e}")

    # Migration: prediction_audit needs final_prediction (REAL/FAKE binary verdict)
    # separate from `verdict` (the 5-level category like "Uncertain"). Accuracy
    # tracking compares against the binary verdict, not the category.
    try:
        cursor.execute("PRAGMA table_info(prediction_audit)")
        columns = [row[1] for row in cursor.fetchall()]
        if "final_prediction" not in columns:
            cursor.execute("ALTER TABLE prediction_audit ADD COLUMN final_prediction TEXT")
            print("🔧 Database migration: added 'final_prediction' to prediction_audit table.")
    except Exception as e:
        print(f"⚠️ Database migration error: {e}")

    # Migration: feedback needs article_hash to reliably join back to the exact
    # prediction_audit row it's confirming/correcting, instead of matching on
    # raw text (fragile across whitespace/casing differences and duplicates).
    try:
        cursor.execute("PRAGMA table_info(feedback)")
        columns = [row[1] for row in cursor.fetchall()]
        if "article_hash" not in columns:
            cursor.execute("ALTER TABLE feedback ADD COLUMN article_hash TEXT")
            print("🔧 Database migration: added 'article_hash' to feedback table.")
    except Exception as e:
        print(f"⚠️ Database migration error: {e}")
        
    # Prepopulate tables if empty
    try:
        cursor.execute("SELECT COUNT(*) FROM source_reputation")
        if cursor.fetchone()[0] == 0:
            from utils.source_engine import SourceEngine
            se = SourceEngine()
            for domain, info in se.reputation_db.items():
                cursor.execute("""
                    INSERT OR IGNORE INTO source_reputation (domain, trust_score, bias, category, description, fact_check_history)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    domain,
                    float(info.get("score", 50.0)),
                    info.get("bias", "Unknown"),
                    info.get("category", "Unknown"),
                    info.get("notes", ""),
                    "Initial setup."
                ))
    except Exception:
        pass

    try:
        cursor.execute("SELECT COUNT(*) FROM misinformation_patterns")
        if cursor.fetchone()[0] == 0:
            default_patterns = [
                ("Health Myths", r"\b(cure for cancer|vaccines cause autism|5g radiation covid|miracle juice|mms drops|nano-chips)\b", 0.3),
                ("Election Misinformation", r"\b(ballot harvesting|stolen election|rigged voting machines|dead people voted|faked ballots)\b", 0.4),
                ("Financial Scams", r"\b(guaranteed double returns|elon musk crypto giveaway|make 10000 daily|get rich quick scheme|whatsapp cash gift)\b", 0.3),
                ("Conspiracy Theories", r"\b(flat earth|illuminati secret society|chem-trails control|reptilian shape-shifter|deep state cabal)\b", 0.3)
            ]
            for theme, pattern, multiplier in default_patterns:
                cursor.execute("""
                    INSERT INTO misinformation_patterns (theme, regex_pattern, risk_multiplier)
                    VALUES (?, ?, ?)
                """, (theme, pattern, multiplier))
    except Exception:
        pass
        
    conn.commit()
    conn.close()

# ------------------------------------------------------------------
# Model & Helper Initialization
# ------------------------------------------------------------------

def load_services():
    """Instantiate and load ML models and helper utilities."""
    global model, preprocessor, clickbait_detector, ai_detector
    global claim_verifier, multilingual_processor, source_engine, stance_detector, last_loaded_mtime
    
    # Initialize utility modules
    preprocessor = TextPreprocessor()
    clickbait_detector = ClickbaitDetector()
    ai_detector = AIContentDetector()
    claim_verifier = ClaimVerifier()
    multilingual_processor = MultilingualProcessor()
    source_engine = SourceEngine()
    stance_detector = StanceDetector()
    
    # Initialize DB tables
    try:
        init_db()
    except Exception as e:
        print(f"⚠️ Failed database init: {e}")

    # Load Model Pipeline
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            last_loaded_mtime = os.path.getmtime(model_path)
            print(f"✅ Loaded VotingClassifier pipeline from {model_path}")
        except Exception as e:
            print(f"⚠️ Error loading model file: {e}")
            model = None
    else:
        print("⚠️ Model pkl not found. Running in rule-based fallback mode.")
        model = None

def check_and_reload_model():
    """Reload model if the file on disk has changed since last load."""
    global model, last_loaded_mtime
    if os.path.exists(model_path):
        try:
            mtime = os.path.getmtime(model_path)
            if mtime > last_loaded_mtime:
                print("🔄 Model file change detected on disk. Reloading...")
                model = joblib.load(model_path)
                last_loaded_mtime = mtime
                print("✅ Model reloaded successfully.")
        except Exception as e:
            print(f"⚠️ Error checking or reloading model: {e}")

# ------------------------------------------------------------------
# Core Prediction Logic
# ------------------------------------------------------------------

def predict_article(text, model, preprocessor, clickbait_detector=None, ai_detector=None, claim_verifier=None, source_engine=None, url=None, article_hash_override=None):
    """
    Advanced credibility analysis decision engine.

    article_hash_override: if provided, used as the audit-log article_hash
    instead of hashing `text` directly. Callers that translate text before
    calling this function (see /predict, /analyze) should pass the hash of
    the ORIGINAL, pre-translation text here, so the same article always
    gets the same hash regardless of language — otherwise a non-English
    article's audit-log hash would never match the hash /feedback computes
    from the user's original submission, and feedback would silently fail
    to join against its prediction in /monitoring/accuracy.
    """
    import hashlib
    import json
    import sqlite3
    
    check_and_reload_model()
    
    text_hash = article_hash_override if article_hash_override else hashlib.md5(text.encode('utf-8')).hexdigest()
    processed = preprocessor.preprocess_for_model(text)
    
    try:
        prediction = model.predict([processed])[0]
        probs = model.predict_proba([processed])[0]
        classes = list(model.classes_)
        raw_confidence = float(probs[classes.index(prediction)])
        # ml_score = P(REAL) directly from the model — this is the true signal
        real_idx = classes.index('REAL') if 'REAL' in classes else 1
        ml_score = float(probs[real_idx])
    except Exception:
        prediction = 'REAL'
        raw_confidence = 0.55
        probs = [0.5, 0.5]
        classes = ['FAKE', 'REAL']
        ml_score = 0.5
        
    # ── Transformer Classification (always-on when available) ──
    bert_triggered = False
    bert_result = None
    transformer_used = False
    transformer_model = "N/A"
    transformer_score = None
    try:
        from utils.transformer_predictor import TransformerPredictor
        _transformer_predictor = TransformerPredictor()
        if _transformer_predictor.is_available:
            _lang = "en"
            try:
                from utils.multilingual import MultilingualProcessor
                _mp = MultilingualProcessor()
                _det = _mp.detect_language(text)
                _lang = _det.get("language_code", "en")
            except Exception:
                pass
            t_res = _transformer_predictor.predict(text, language_code=_lang)
            if t_res:
                transformer_used = True
                bert_triggered = True
                transformer_model = t_res.get("model_used", "transformer")
                bert_result = t_res
                transformer_score = t_res["probabilities"]["REAL"]
                ml_score = (ml_score * 0.35 + transformer_score * 0.65)
                if ml_score >= 0.5:
                    prediction = 'REAL'
                    raw_confidence = ml_score
                else:
                    prediction = 'FAKE'
                    raw_confidence = 1.0 - ml_score
    except Exception:
        pass
            
    # ── Short-text penalty: push ml_score toward 0.5 (uncertain) for short inputs ──
    word_count = len(text.split())
    if word_count < 150:
        length_factor = max(0.5, word_count / 150.0)
        raw_confidence *= length_factor
        # Shrink ml_score toward 0.5 for short text — model is unreliable on snippets
        ml_score = 0.5 + (ml_score - 0.5) * length_factor
        
    indicators = preprocessor.analyze_suspicious_indicators(text)
    sensationalism = indicators.get('sensationalism_score', 0.0)
    cred_signal = indicators.get('credibility_score', 0.0)
    nlp_nudge = (cred_signal - sensationalism) * 0.5
    nlp_score = max(0.0, min(1.0, 0.5 + nlp_nudge))
    
    if clickbait_detector is None:
        from utils.clickbait_detector import ClickbaitDetector
        clickbait_detector = ClickbaitDetector()
    clickbait_res = clickbait_detector.detect(text, title=(url if url else text[:100]))
    clickbait_score = float(clickbait_res.get("clickbait_score", 0.0))
    
    if ai_detector is None:
        from utils.ai_detector import AIContentDetector
        ai_detector = AIContentDetector()
    ai_res = ai_detector.detect(text)
    ai_score = float(ai_res.get("ai_score", 0.0))
    
    if source_engine is None:
        from utils.source_engine import SourceEngine
        source_engine = SourceEngine()
    # ── Source Trust: only use actual URLs, never parse article text as domain ──
    source_trust = 50.0
    source_profile = None
    has_valid_url = bool(url and url.strip() and ('.' in url) and len(url.strip()) < 500)
    
    if has_valid_url:
        domain_to_check = url
        try:
            db_path_src = os.path.join(PROJECT_ROOT, "data", "truthshield.db")
            if os.path.exists(db_path_src):
                conn = sqlite3.connect(db_path_src)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                domain = source_engine.clean_domain(domain_to_check)
                cursor.execute("SELECT * FROM source_reputation WHERE domain = ?", (domain,))
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    source_trust = float(row["trust_score"])
                    source_profile = {
                        "domain": row["domain"],
                        "score": source_trust,
                        "category": row["category"],
                        "bias": row["bias"],
                        "description": row["description"]
                    }
        except Exception:
            pass
        
        if not source_profile:
            source_profile = source_engine.get_trust_profile(domain_to_check)
            source_trust = float(source_profile.get("score", 50.0))
    else:
        # No URL provided — use neutral defaults, do NOT parse article text as a domain
        source_profile = {
            "domain": "(no URL provided)",
            "score": 50.0,
            "category": "Unknown",
            "bias": "Unknown",
            "description": "No source URL was provided for domain reputation analysis."
        }
        
    if claim_verifier is None:
        from utils.claim_verifier import ClaimVerifier
        claim_verifier = ClaimVerifier()
    try:
        verification_res = claim_verifier.verify_article(text)
        verification_results = verification_res.get("verification_results", [])
        verification_status = verification_res.get("summary", "Unverified claims.")
        factcheck_score = float(verification_res.get("overall_verification_score", 0.5))
        evidence_count = int(verification_res.get("evidence_count", 0))
        evidence_quality = float(verification_res.get("evidence_quality", 0.0))
        agreement_ratio = float(verification_res.get("agreement_ratio", 0.5))
        temporal_analysis = verification_res.get("temporal_analysis", {"is_consistent": True, "risk_score": 0.0, "mismatches": []})
    except Exception:
        verification_res = {}
        verification_results = []
        verification_status = "Fact verification unavailable."
        factcheck_score = 0.5
        evidence_count = 0
        evidence_quality = 0.0
        agreement_ratio = 0.5
        temporal_analysis = {"is_consistent": True, "risk_score": 0.0, "mismatches": []}
        
    # ── Satire Detection ──
    try:
        from utils.satire_detector import SatireDetector
        satire_detector = SatireDetector()
        satire_res = satire_detector.detect(text, url=url)
    except Exception:
        satire_res = {"satire_score": 0.0, "is_satire": False}
    
    satire_score = satire_res.get("satire_score", 0.0)
    is_satire = satire_res.get("is_satire", False)

    # ── Stance Detection ──
    try:
        from utils.stance_detector import StanceDetector
        local_stance_detector = StanceDetector()
        stance_claims = verification_res.get("claims", [])
    except Exception:
        local_stance_detector = None
        stance_claims = []
        
    if local_stance_detector:
        stance_result = local_stance_detector.detect(text, claims=stance_claims,
                                                     verification_results=verification_results)
    elif stance_detector:
        stance_result = stance_detector.detect(text, claims=stance_claims,
                                               verification_results=verification_results)
    else:
        stance_result = {
            "stance": "NEUTRAL",
            "stance_confidence": 0.5,
            "refutation_signals": [],
            "support_signals": [],
            "factchecker_references": [],
            "attribution_count": 0,
            "is_factcheck_article": False,
            "scores": {"refutation": 0.0, "support": 0.0}
        }
    article_stance = stance_result["stance"]
    stance_confidence = stance_result["stance_confidence"]
    is_factcheck_article = stance_result["is_factcheck_article"]
    
    # Only treat the article as REFUTES if it is verified as a factcheck article.
    # This prevents regular fake news articles that quote sources or mention WHO/scientists
    # from hijacking the refutation logic and bypassing red-flag penalties.
    if article_stance == "REFUTES" and not is_factcheck_article:
        article_stance = "NEUTRAL"
        stance_confidence = 0.5


    # ── Misinformation Pattern Matching ──
    matched_themes = []
    misinfo_multiplier = 1.0
    try:
        db_path = os.path.join(PROJECT_ROOT, "data", "truthshield.db")
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT theme, regex_pattern, risk_multiplier FROM misinformation_patterns")
            rows = cursor.fetchall()
            conn.close()
            
            for row in rows:
                theme = row["theme"]
                pattern = row["regex_pattern"]
                multiplier = float(row["risk_multiplier"])
                
                # Check for match in text
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    matched_themes.append({
                        "theme": theme,
                        "match": match.group(0),
                        "multiplier": multiplier
                    })
                    # If the article is REFUTING the misinformation, do NOT penalize it
                    if article_stance != "REFUTES":
                        misinfo_multiplier *= (1.0 - multiplier)
    except Exception:
        pass

    # ── 9. Weighted Ensemble Decision Engine (Transformer-Primary, Stance-Aware) ──
    has_source_signal = has_valid_url and source_profile is not None and source_profile.get("category") not in ["Unknown", "Unverified Source"]
    has_factcheck_signal = evidence_count > 0
    
    if transformer_used:
        w_ml = 0.40
        w_source = 0.15
        w_factcheck = 0.20
        w_clickbait = 0.05
        w_nlp = 0.05
        w_ai = 0.05
        w_satire = 0.05
        w_sentiment = 0.05
    else:
        w_ml = 0.31
        w_source = 0.20
        w_factcheck = 0.25
        w_clickbait = 0.05
        w_nlp = 0.06
        w_ai = 0.05
        w_satire = 0.08
        w_sentiment = 0.0
    
    if not has_source_signal:
        w_ml += w_source * 0.5
        w_nlp += w_source * 0.5
        w_source = 0.0
    if not has_factcheck_signal:
        w_ml += w_factcheck * 0.7
        w_nlp += w_factcheck * 0.3
        w_factcheck = 0.0
    if not has_source_signal and not has_factcheck_signal:
        w_nlp += 0.05
        w_ml -= 0.05
    
    sentiment_bias = 1.0 - indicators.get("sensationalism_score", 0.0)
    
    credibility = (
        ml_score * w_ml +
        (source_trust / 100.0) * w_source +
        factcheck_score * w_factcheck +
        (1.0 - clickbait_score) * w_clickbait +
        nlp_score * w_nlp +
        (1.0 - ai_score) * w_ai +
        (1.0 - satire_score) * w_satire +
        sentiment_bias * w_sentiment
    )

    
    # ── Stance-aware adjustment ──
    if article_stance == "REFUTES" and stance_confidence >= 0.2:
        # The article is debunking/fact-checking a false claim.
        # The ML model may have said FAKE because it sees misinformation keywords
        # (e.g., "hoax", "conspiracy", "vaccines cause autism") but those words
        # appear in the context of refutation, not endorsement.
        
        # 1. Dampen ML penalty: if ML said FAKE, reduce its negative pull
        if prediction == 'FAKE':
            # Flip the ML contribution: a fact-check article containing "fake" keywords is credible
            ml_boost = (1.0 - ml_score) * stance_confidence * 0.35
            credibility += ml_boost
        
        # 2. Boost from refutation evidence
        refutation_boost = stance_confidence * 0.15
        credibility += refutation_boost
        
        # 3. Boost from fact-checker references
        if stance_result.get("factchecker_references"):
            credibility += min(len(stance_result["factchecker_references"]) * 0.05, 0.15)
        
        # 4. Boost from credibility indicators (experts, studies, etc.)
        cred_indicator_boost = cred_signal * 0.1
        credibility += cred_indicator_boost
        
    elif article_stance == "SUPPORTS" and stance_confidence >= 0.3:
        # Article is promoting/endorsing a suspicious claim — apply extra penalty
        credibility -= stance_confidence * 0.1
    
    # ── Red-flag penalty: direct credibility reduction for fabrication indicators ──
    redflag_count = indicators.get('redflag_count', 0)
    if redflag_count > 0 and article_stance != "REFUTES":
        # Each red flag directly penalizes credibility (up to -0.55 total)
        redflag_penalty = min(redflag_count * 0.10, 0.55)
        credibility -= redflag_penalty
        
        # Enforce ceiling overrides for multiple fabrication indicators
        if redflag_count >= 5:
            credibility = min(credibility, 0.18)
        elif redflag_count >= 3:
            credibility = min(credibility, 0.42)
    
    # Apply misinformation patterns multiplier
    credibility = credibility * misinfo_multiplier
    credibility = float(max(0.0, min(1.0, credibility)))
    
    # ── Temporal Staleness Penalty ──
    import datetime
    current_year = datetime.datetime.now().year
    
    stale_years_found = []
    freshness_words = ["latest", "new", "breaking", "just released", "current", "record", "today", "this week"]
    
    # Find all 4-digit years in text
    all_year_matches = list(re.finditer(r'\b(19\d{2}|20\d{2})\b', text))
    
    # Find all freshness word indices
    freshness_indices = []
    text_lower = text.lower()
    for word in freshness_words:
        for m in re.finditer(r'\b' + re.escape(word) + r'\b', text_lower):
            freshness_indices.append(m.start())
            
    # Check if any old year (current_year - 6) is within 80 characters of any freshness word
    stale_count = 0
    for y_match in all_year_matches:
        year_val = int(y_match.group(1))
        if year_val <= (current_year - 6):
            # Check proximity
            y_pos = y_match.start()
            is_stale = False
            for f_pos in freshness_indices:
                if abs(y_pos - f_pos) <= 80:
                    is_stale = True
                    break
            if is_stale:
                stale_years_found.append(year_val)
                stale_count += 1
                
    temporal_penalty = min(stale_count * 0.10, 0.30)
    if temporal_analysis.get("risk_score", 0.0) > 0.3:
        temporal_penalty += 0.10
        
    credibility -= temporal_penalty
    credibility = float(max(0.0, min(1.0, credibility)))
    indicators["temporal_penalty"] = temporal_penalty

    
    # ── 10. Evidence Sufficiency Check ──
    is_sufficient = True
    source_is_known = source_profile.get("category") not in ["Unverified Source", "Unknown"]
    if source_trust <= 55.0 and evidence_count == 0 and article_stance != "REFUTES":
        is_sufficient = False
    elif evidence_count > 0 and evidence_quality < 0.3:
        is_sufficient = False
        
    # ── 11. Reliability Score ──
    source_rel_weight = 1.0 if source_is_known else 0.5
    ev_rel = min(evidence_count / 4.0, 1.0) if evidence_count > 0 else 0.2
    agree_rel = (abs(agreement_ratio - 0.5) * 2.0) if evidence_count > 0 else 0.5
    ml_rel = raw_confidence
    # Stance detection adds to reliability when confident
    stance_rel = stance_confidence if article_stance in ("REFUTES", "SUPPORTS") else 0.3
    
    reliability = (
        source_rel_weight * 0.25 +
        ev_rel * 0.25 +
        agree_rel * 0.15 +
        ml_rel * 0.15 +
        stance_rel * 0.20
    )
    reliability = float(max(0.0, min(1.0, reliability)))
    
    # ── 12. 5-Level Verdict & Insufficient Override ──
    if credibility >= 0.85:
        category = "Highly Credible"
    elif credibility >= 0.65:
        category = "Likely Real"
    elif credibility >= 0.45:
        category = "Uncertain"
    elif credibility >= 0.20:
        category = "Likely Fake"
    else:
        category = "High Risk Misinformation"
        
    # Override: fact-check articles with strong refutation should not be "Uncertain" or worse
    if is_factcheck_article and credibility >= 0.40 and category in ("Uncertain", "Likely Fake", "High Risk Misinformation"):
        category = "Likely Real"
        credibility = max(credibility, 0.65)
        
    # Apply evidence sufficiency fallback (but not for identified fact-check articles or low-credibility/red-flagged articles)
    if not is_sufficient and not is_factcheck_article and credibility >= 0.40:
        category = "Uncertain"
        reliability = min(reliability, 0.40)

        
    # Final adjusted confidence — blend reliability with stance and evidence strength
    evidence_confidence = min(evidence_count / 3.0, 1.0) if evidence_count > 0 else 0.3
    confidence = (
        reliability * 0.5 +
        evidence_confidence * 0.25 +
        stance_confidence * 0.25
    )
    confidence = float(min(max(confidence, 0.05), 0.99))
    
    # Set final binary prediction (dynamic threshold based on NLP score and red flags if no external signals present)
    if not has_source_signal and not has_factcheck_signal:
        base_threshold = 0.55
        # lower threshold when NLP signals strongly suggest fake
        if nlp_score < 0.4:
            threshold = base_threshold - 0.05  # 0.50, easier to call FAKE
        elif redflag_count >= 2:
            threshold = base_threshold - 0.08  # 0.47
        else:
            threshold = base_threshold
    else:
        threshold = 0.50
        
    final_prediction = "REAL" if credibility >= threshold else "FAKE"
    
    # Satire override
    if is_satire and satire_score > 0.65:
        final_prediction = "FAKE"
        category = "Satire / Parody"
        credibility = min(credibility, 0.35)
        
    # Sub-classification flags
    is_clickbait = clickbait_score > 0.6
    is_ai_generated = ai_score > 0.75
    is_satire = is_satire or (source_profile.get("category") in ["Satire / Parody", "Parody"] if source_profile else False)

    
    # Risk Factor Breakdown
    positive_factors = []
    negative_factors = []
    
    # ── Stance-based factors (highest priority) ──
    if article_stance == "REFUTES" and stance_confidence >= 0.2:
        refutation_sigs = ", ".join(stance_result["refutation_signals"][:3]) if stance_result["refutation_signals"] else "refutation language detected"
        positive_factors.append({"factor": "Fact-Check / Debunking Article", "detail": f"Article refutes claims with evidence ({refutation_sigs}).", "impact": f"+{int(stance_confidence * 30)}%"})
        if stance_result.get("factchecker_references"):
            refs = ", ".join(stance_result["factchecker_references"][:3])
            positive_factors.append({"factor": "Fact-Checker References", "detail": f"Cites fact-checking sources: {refs}.", "impact": "+15%"})
    elif article_stance == "SUPPORTS" and stance_confidence >= 0.3:
        support_sigs = ", ".join(stance_result["support_signals"][:3]) if stance_result["support_signals"] else "promotional language"
        negative_factors.append({"factor": "Promotes Unverified Claims", "detail": f"Article endorses claims without evidence ({support_sigs}).", "impact": f"-{int(stance_confidence * 20)}%"})
        
    if source_trust >= 75.0:
        positive_factors.append({"factor": "Trusted Publisher", "detail": f"Source {source_profile['domain']} has high trust ({source_trust}%).", "impact": "+20%"})
    elif source_trust <= 40.0:
        negative_factors.append({"factor": "Untrusted Source", "detail": f"Source {source_profile['domain']} is flagged as low-trust ({source_trust}%).", "impact": "-20%"})
        
    for theme_match in matched_themes:
        # If article refutes the misinformation, show it as informational, not a penalty
        if article_stance == "REFUTES":
            positive_factors.append({
                "factor": f"Addresses: {theme_match['theme']}",
                "detail": f"Article discusses and debunks {theme_match['theme'].lower()} topic (\"{theme_match['match']}\").",
                "impact": "+5%"
            })
        else:
            negative_factors.append({
                "factor": f"Misinformation: {theme_match['theme']}",
                "detail": f"Content matches known pattern for {theme_match['theme'].lower()} (\"{theme_match['match']}\").",
                "impact": f"-{int(theme_match['multiplier'] * 100)}%"
            })

    # Red-flag pattern matches
    redflags_list = indicators.get('redflags', [])
    if redflags_list and article_stance != "REFUTES":
        rf_examples = ", ".join(f'"{rf}"' for rf in redflags_list[:3])
        negative_factors.append({
            "factor": "Fabrication Indicators",
            "detail": f"Detected {len(redflags_list)} red-flag phrases: {rf_examples}.",
            "impact": f"-{min(len(redflags_list) * 6, 30)}%"
        })

    if evidence_count > 0:
        if agreement_ratio >= 0.75:
            positive_factors.append({"factor": "Fact Matches", "detail": f"High RAG evidence agreement ({agreement_ratio * 100:.0f}% supporting).", "impact": "+25%"})
        elif agreement_ratio <= 0.25:
            negative_factors.append({"factor": "Contradicted Claims", "detail": f"RAG checks find direct contradictions in claims.", "impact": "-25%"})
            
    if not temporal_analysis.get("is_consistent", True):
        negative_factors.append({"factor": "Temporal Drift", "detail": "Reused old statistics or outdated event timeline detected.", "impact": f"-{int(temporal_analysis['risk_score'] * 30)}%"})
        
    if is_clickbait:
        negative_factors.append({"factor": "Clickbait Headlines", "detail": f"Sensationalized framing detected ({clickbait_score * 100:.0f}% clickbait).", "impact": "-5%"})
    else:
        positive_factors.append({"factor": "Editorial Title", "detail": "Neutral, objective title formatting.", "impact": "+5%"})
        
    if is_ai_generated:
        negative_factors.append({"factor": "Linguistic Uniformity", "detail": f"High similarity to AI-generated text ({ai_score * 100:.0f}% AI score).", "impact": "-5%"})
        
    if raw_confidence > 0.75 and prediction == 'REAL':
        positive_factors.append({"factor": "Model Signal", "detail": f"Classifier strongly validates text patterns.", "impact": "+35%"})
    elif raw_confidence > 0.75 and prediction == 'FAKE' and article_stance != "REFUTES":
        negative_factors.append({"factor": "Stylistic Flags", "detail": f"Classifier detects typical misinformation patterns.", "impact": "-35%"})
    elif raw_confidence > 0.75 and prediction == 'FAKE' and article_stance == "REFUTES":
        positive_factors.append({"factor": "Misinformation Keywords (Debunking Context)", "detail": "Classifier detected misinformation-related keywords, but article uses them in a refutation/fact-check context.", "impact": "+10%"})
        
    # ── 13. Audit Log & Drift Monitoring ──
    try:
        db_path = os.path.join(PROJECT_ROOT, "data", "truthshield.db")
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Calculate dynamic distributions
            cursor.execute("SELECT COUNT(*) FROM feedback")
            feedback_count = cursor.fetchone()[0]
            
            monthly_accuracy = 0.90 # default
            if feedback_count and feedback_count > 0:
                cursor.execute("SELECT SUM(CASE WHEN user_verdict = 'Agree with AI' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) FROM feedback")
                accuracy_val = cursor.fetchone()[0]
                if accuracy_val is not None:
                    monthly_accuracy = float(accuracy_val)

            # Get prediction distribution for last 30 days
            cursor.execute("""
                SELECT verdict, COUNT(*) 
                FROM prediction_audit 
                WHERE timestamp >= datetime('now', '-30 days')
                GROUP BY verdict
            """)
            verdict_counts = dict(cursor.fetchall())
            verdict_counts[category] = verdict_counts.get(category, 0) + 1
            
            # Get source distribution (trust score categories)
            cursor.execute("""
                SELECT CAST(source_score AS INTEGER), COUNT(*) 
                FROM prediction_audit 
                WHERE timestamp >= datetime('now', '-30 days')
                GROUP BY CAST(source_score AS INTEGER)
            """)
            source_dist = {str(k): v for k, v in cursor.fetchall()}
            source_trust_int = int(source_trust)
            source_dist[str(source_trust_int)] = source_dist.get(str(source_trust_int), 0) + 1
            
            feat_contrib = {
                "ml_model": round(w_ml, 4),
                "source_trust": round(w_source, 4),
                "fact_check": round(w_factcheck, 4),
                "nlp_indicators": round(w_nlp, 4),
                "clickbait": round(w_clickbait, 4),
                "ai_generated": round(w_ai, 4),
                "satire": round(w_satire, 4),
            }
            cursor.execute("""
                INSERT INTO prediction_audit (
                    article_hash, verdict, final_prediction, confidence, credibility, reliability, 
                    source_score, factcheck_score, clickbait_score, ai_score, 
                    features_contribution, monthly_accuracy, source_distribution, prediction_distribution
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                text_hash, category, final_prediction, confidence, credibility, reliability,
                source_trust, factcheck_score, clickbait_score, ai_score,
                json.dumps(feat_contrib), monthly_accuracy, json.dumps(source_dist), json.dumps(verdict_counts)
            ))
            conn.commit()
            conn.close()
    except Exception:
        pass
        
    # Map to 3-tier display verdict
    if credibility >= 0.65:
        display_verdict = "\U0001f7e2 Likely True"
    elif credibility >= 0.45:
        display_verdict = "\U0001f7e1 Needs Verification"
    else:
        display_verdict = "\U0001f534 Likely Misleading"
    
    # Generate evidence report
    evidence_report = None
    try:
        from utils.evidence_engine import EvidenceEngine
        _ee = EvidenceEngine()
        _partial = {
            'prediction': final_prediction, 'confidence': confidence,
            'credibility': credibility, 'reliability': reliability,
            'category': category, 'clickbait_score': clickbait_score,
            'ai_score': ai_score, 'source_trust': source_trust,
            'source_profile': source_profile, 'verification_results': verification_results,
            'indicators': indicators, 'bert_triggered': bert_triggered,
            'is_clickbait': is_clickbait, 'is_ai_generated': is_ai_generated,
            'is_satire': is_satire, 'nlp_score': nlp_score,
            'factcheck_score': factcheck_score, 'article_stance': article_stance,
            'stance_confidence': stance_confidence, 'ml_score': ml_score,
            'evidence_count': evidence_count, 'matched_themes': matched_themes,
            'satire_score': satire_score, 'temporal_analysis': temporal_analysis,
            'stance': stance_result, 'transformer_used': transformer_used,
            'transformer_model': transformer_model,
        }
        evidence_report = _ee.generate_report(_partial)
    except Exception:
        evidence_report = None
    
    return {
        'prediction': final_prediction,
        'article_hash': text_hash,
        'confidence': confidence,
        'credibility': credibility,
        'reliability': reliability,
        'category': category,
        'display_verdict': display_verdict,
        'clickbait_score': clickbait_score,
        'ai_score': ai_score,
        'source_trust': source_trust,
        'source_profile': source_profile,
        'verification_results': verification_results,
        'verification_status': verification_status,
        'indicators': indicators,
        'bert_triggered': bert_triggered,
        'bert_result': bert_result,
        'transformer_used': transformer_used,
        'transformer_model': transformer_model,
        'is_sufficient': is_sufficient,
        'is_clickbait': is_clickbait,
        'is_ai_generated': is_ai_generated,
        'is_satire': is_satire,
        'positive_factors': positive_factors,
        'negative_factors': negative_factors,
        'temporal_analysis': temporal_analysis,
        'stance': stance_result,
        'ml_score': ml_score,
        'nlp_score': nlp_score,
        'factcheck_score': factcheck_score,
        'article_stance': article_stance,
        'stance_confidence': stance_confidence,
        'evidence_count': evidence_count,
        'matched_themes': matched_themes,
        'satire_score': satire_score,
        'evidence_report': evidence_report,
    }

@app.on_event("startup")
async def on_startup():
    """
    Ensure services are loaded whenever the app starts, regardless of how
    it's launched (e.g. `uvicorn api_server:app`, gunicorn workers, etc.),
    not just the `python api_server.py` path below.
    """
    if model is None and preprocessor is None:
        print("🚀 Initializing TruthShield API services (startup event)...")
        load_services()

# ------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------

# ── Image Analysis Endpoint (Phase 5) ──

class ImageAnalyzeRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image")
    run_ocr: bool = True
    check_manipulation: bool = True
    reverse_search: bool = False

@app.post("/analyze-image")
async def analyze_image(request: ImageAnalyzeRequest):
    """Analyze an image for OCR text, manipulation, and provenance."""
    try:
        import base64
        image_bytes = base64.b64decode(request.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    try:
        from utils.image_verifier import ImageVerifier
        verifier = ImageVerifier()
        result = verifier.verify_image(
            image_bytes,
            run_ocr=request.run_ocr,
            check_manipulation=request.check_manipulation,
            reverse_search=request.reverse_search,
        )

        # If OCR extracted text, also run text prediction
        text_prediction = None
        ocr_text = ""
        if result.get("ocr") and result["ocr"].get("text"):
            ocr_text = result["ocr"]["text"]
            if len(ocr_text) >= 20 and model and preprocessor:
                text_prediction = predict_article(
                    ocr_text, model, preprocessor,
                    clickbait_detector, ai_detector,
                    claim_verifier, source_engine,
                )

        return {
            "status": "ok",
            "image_analysis": result,
            "ocr_text": ocr_text,
            "text_prediction": text_prediction,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

# ── Full Pipeline Endpoint (Phase 4) ──

class PipelineRequest(BaseModel):
    text: str = Field(..., min_length=20, description="Article text to analyze")
    url: Optional[str] = None

@app.post("/pipeline")
async def run_pipeline(request: PipelineRequest):
    """Run the full verification pipeline with stage tracking."""
    try:
        from utils.pipeline import VerificationPipeline
        pipeline = VerificationPipeline()
        result = pipeline.run(request.text, url=request.url)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "TruthShield Authenticity Engine API is online.",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_path": model_path,
        "model_loaded": model is not None,
        "db_connected": os.path.exists(db_path)
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Simple prediction endpoint compatible with v1."""
    import hashlib
    text = request.text
    article_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    
    # Translate multilingual content. multilingual_processor is set by
    # load_services() via the startup event before any request can reach
    # this line; Pylance can't trace that cross-function reassignment.
    res_lang = multilingual_processor.process_for_analysis(text)  # type: ignore[union-attr]
    eng_text = res_lang["processed_text"]
    
    # 2. Get predictions using unified pipeline
    res = predict_article(
        eng_text, model, preprocessor, clickbait_detector, ai_detector,
        claim_verifier, source_engine, request.url,
        article_hash_override=article_hash
    )
    
    return PredictResponse(
        prediction=res['prediction'],
        article_hash=res['article_hash'],
        confidence=res['confidence'],
        credibility=res['credibility'],
        reliability=res['reliability'],
        summary=res['verification_status'],
        is_fake=(res['prediction'] == "FAKE")
    )

@app.post("/predict-json")
async def predict_json(request: PredictRequest):
    """Direct JSON fallback compatible with browser extension."""
    try:
        res = await predict(request)
        return JSONResponse(content=res.model_dump())
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Advanced multi-class deep analysis endpoint.
    Runs ML predictions, clickbait scoring, AI content checks, and fact verification.
    """
    import hashlib
    text = request.text
    url = request.url or ""
    article_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    
    # Multilingual processing. multilingual_processor is set by load_services()
    # via the startup event before any request can reach this line; Pylance
    # can't trace that cross-function reassignment.
    res_lang = multilingual_processor.process_for_analysis(text)  # type: ignore[union-attr]
    eng_text = res_lang["processed_text"]
    detected_lang_name = res_lang["language_name"]
    was_translated = res_lang["was_translated"]
    
    # 2. Run prediction pipeline
    res = predict_article(
        eng_text, model, preprocessor, clickbait_detector, ai_detector,
        claim_verifier, source_engine, url,
        article_hash_override=article_hash
    )
    
    return AnalyzeResponse(
        prediction=res['prediction'],
        article_hash=res['article_hash'],
        confidence=res['confidence'],
        credibility=res['credibility'],
        reliability=res['reliability'],
        category=res['category'],
        clickbait_score=res['clickbait_score'],
        ai_score=res['ai_score'],
        source_trust=res['source_trust'],
        verification_results=res['verification_results'],
        summary=res['verification_status'],
        language=detected_lang_name,
        was_translated=was_translated
    )

@app.post("/feedback")
async def save_feedback(request: FeedbackRequest):
    """Store explicit user correctness feedback in the DB."""
    try:
        import hashlib
        article_hash = hashlib.md5(request.text.encode('utf-8')).hexdigest()

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedback (user_email, text, model_prediction, user_verdict, rating, notes, article_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            request.user_email, request.text, request.model_prediction,
            request.user_verdict, request.rating, request.notes, article_hash
        ))
        conn.commit()
        conn.close()
        return {"status": "ok", "message": "Feedback submitted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/history")
async def get_history(limit: int = 50):
    """Retrieve recent predictions and analyses from the history database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, user_email, title, prediction, confidence, credibility, category, timestamp FROM history ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        history_list = []
        for r in rows:
            history_list.append({
                "id": r["id"],
                "user_email": r["user_email"],
                "title": r["title"],
                "prediction": r["prediction"],
                "confidence": r["confidence"],
                "credibility": r["credibility"],
                "category": r["category"] if r["category"] else r["prediction"],
                "timestamp": r["timestamp"]
            })
        return history_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/monitoring/accuracy")
async def get_monitoring_accuracy(days: int = 30):
    """
    Live accuracy tracking: joins every logged prediction (prediction_audit,
    written automatically on every /predict and /analyze call) against any
    user feedback received for that same article (feedback.article_hash),
    over a rolling window.

    Accuracy here means: of the predictions where a user later confirmed or
    corrected the verdict, what fraction did the model get right? This is a
    proxy for true accuracy (not every prediction gets feedback), not a
    ground-truth label — it reflects what users tell us, same as the existing
    monthly_accuracy figure, but broken out over time instead of one rolling
    number.

    Returns:
      - total_predictions: total predictions logged in the window
      - feedback_coverage: predictions in the window that received feedback
      - overall_accuracy: agreement rate across all feedback in the window
      - daily: per-day breakdown of predictions, feedback count, accuracy
      - verdict_breakdown: accuracy split by the model's final_prediction (REAL vs FAKE)
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM prediction_audit WHERE timestamp >= datetime('now', ?)",
            (f"-{days} days",)
        )
        total_predictions = cursor.fetchone()[0]

        # Join on article_hash: for each feedback row, find the most recent
        # prediction_audit row for that same article within the window.
        cursor.execute("""
            SELECT
                pa.final_prediction,
                f.model_prediction,
                f.user_verdict,
                date(pa.timestamp) as pred_date
            FROM feedback f
            JOIN prediction_audit pa ON pa.article_hash = f.article_hash
            WHERE pa.timestamp >= datetime('now', ?)
              AND f.article_hash IS NOT NULL
        """, (f"-{days} days",))
        joined_rows = cursor.fetchall()
        conn.close()

        def is_agreement(final_prediction, user_verdict):
            """
            user_verdict may be the literal sentinel 'Agree with AI', or it may
            be the user's own corrected label (e.g. 'REAL'/'FAKE'). Handle both
            conventions rather than assume one, since the column has no schema
            constraint and we can't be certain which the frontend sends.
            """
            if user_verdict is None:
                return None
            uv = user_verdict.strip().lower()
            if uv == "agree with ai":
                return True
            if uv in ("disagree with ai", "disagree"):
                return False
            if uv in ("real", "fake") and final_prediction:
                return uv == final_prediction.strip().lower()
            return None  # unrecognized value — exclude rather than guess

        daily = {}
        verdict_breakdown = {"REAL": {"agree": 0, "disagree": 0}, "FAKE": {"agree": 0, "disagree": 0}}
        total_feedback = 0
        total_agree = 0

        for row in joined_rows:
            final_pred = row["final_prediction"]
            user_verdict = row["user_verdict"]
            pred_date = row["pred_date"]

            agreement = is_agreement(final_pred, user_verdict)
            if agreement is None:
                continue

            total_feedback += 1
            if agreement:
                total_agree += 1

            daily.setdefault(pred_date, {"feedback_count": 0, "agree_count": 0})
            daily[pred_date]["feedback_count"] += 1
            if agreement:
                daily[pred_date]["agree_count"] += 1

            if final_pred in verdict_breakdown:
                key = "agree" if agreement else "disagree"
                verdict_breakdown[final_pred][key] += 1

        daily_list = []
        for d in sorted(daily.keys()):
            fc = daily[d]["feedback_count"]
            ac = daily[d]["agree_count"]
            daily_list.append({
                "date": d,
                "feedback_count": fc,
                "accuracy": round(ac / fc, 4) if fc > 0 else None
            })

        overall_accuracy = round(total_agree / total_feedback, 4) if total_feedback > 0 else None

        def verdict_accuracy(v):
            a = verdict_breakdown[v]["agree"]
            d = verdict_breakdown[v]["disagree"]
            total = a + d
            return round(a / total, 4) if total > 0 else None

        return {
            "window_days": days,
            "total_predictions": total_predictions,
            "feedback_coverage": total_feedback,
            "feedback_coverage_pct": round(total_feedback / total_predictions, 4) if total_predictions > 0 else None,
            "overall_accuracy": overall_accuracy,
            "verdict_breakdown": {
                "REAL": {**verdict_breakdown["REAL"], "accuracy": verdict_accuracy("REAL")},
                "FAKE": {**verdict_breakdown["FAKE"], "accuracy": verdict_accuracy("FAKE")},
            },
            "daily": daily_list,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Initializing TruthShield API services...")
    load_services()
    
    port = 8000
    print(f"🌐 Starting production API server on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)