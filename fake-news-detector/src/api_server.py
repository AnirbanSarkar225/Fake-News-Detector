"""
API Server for TruthShield Fake News Detector.

Provides REST endpoints for predictions, advanced analyses, user feedback,
and historical search. Integrates with the database, clickbait detector,
AI content detector, claim verifier, multilingual translator, and source trust engine.
"""

import os
import sys
import io
import json
import sqlite3
import hashlib
import datetime
import threading
import time
from typing import Dict, List, Optional

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import joblib

# Resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

def background_update_task():
    """Background task to periodically fetch updates and retrain the model."""
    while True:
        try:
            # Import the update function from the script
            from scripts.realtime_update import fetch_and_update_model
            fetch_and_update_model()
        except Exception as e:
            print(f"⚠️ Error in background update task: {e}")
        # Sleep for 30 minutes (1800 seconds)
        time.sleep(1800)

# Imports from utils
from utils.preprocess import TextPreprocessor
from utils.clickbait_detector import ClickbaitDetector
from utils.ai_detector import AIContentDetector
from utils.claim_verifier import ClaimVerifier
from utils.multilingual import MultilingualProcessor
from utils.source_engine import SourceEngine
from utils.stance_detector import StanceDetector

# Global variables (initialized in startup event)
model = None
preprocessor = None
clickbait_detector = None
ai_detector = None
claim_verifier = None
multilingual_processor = None
source_engine = None
stance_detector = None

# Pydantic models
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

class ImageAnalyzeRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image")
    run_ocr: bool = True
    check_manipulation: bool = True
    reverse_search: bool = False

class PipelineRequest(BaseModel):
    text: str = Field(..., min_length=20, description="Article text to analyze")
    url: Optional[str] = None

model_path = os.path.join(PROJECT_ROOT, "model", "fake_news_model.pkl")
db_path = os.path.join(PROJECT_ROOT, "data", "truthshield.db")
last_loaded_mtime = 0.0

# FastAPI app
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


def get_db_connection():
    """Get a connection to the SQLite database."""
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
            final_prediction TEXT,
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
            # Insert some default known sources
            default_sources = [
                ("reuters.com", 90.0, "Center", "Reuters", "International news organization", "[]", 0.9, 100),
                ("apnews.com", 88.0, "Center", "Associated Press", "US news cooperative", "[]", 0.88, 100),
                ("bbc.com", 85.0, "Center", "BBC", "British public service broadcaster", "[]", 0.85, 100),
                ("nytimes.com", 80.0, "Lean Left", "New York Times", "US daily newspaper", "[]", 0.8, 100),
                ("foxnews.com", 55.0, "Lean Right", "Fox News", "US conservative news channel", "[]", 0.55, 100),
                ("breitbart.com", 30.0, "Right", "Breitbart News", "US conservative news website", "[]", 0.3, 100),
            ]
            cursor.executemany("""
                INSERT OR IGNORE INTO source_reputation
                (domain, trust_score, bias, category, description, fact_check_history, accuracy_rate, frequency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, default_sources)
            print("🔧 Prepopulated source_reputation table with default sources.")
    except Exception as e:
        print(f"⚠️ Error prepopulating source_reputation: {e}")

    conn.commit()
    conn.close()


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


@app.on_event("startup")
async def on_startup():
    """
    Initialize services and load models on startup.
    """
    global model, preprocessor, clickbait_detector, ai_detector, claim_verifier, \
           multilingual_processor, source_engine, stance_detector

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

    # Initialize services (lazy-loaded where possible)
    try:
        preprocessor = TextPreprocessor()
        clickbait_detector = ClickbaitDetector()
        ai_detector = AIContentDetector()
        claim_verifier = ClaimVerifier()
        multilingual_processor = MultilingualProcessor()
        source_engine = SourceEngine()
        stance_detector = StanceDetector()
    except Exception as e:
        print(f"⚠️ Error initializing services: {e}")

    # Start background update thread
    update_thread = threading.Thread(target=background_update_task, daemon=True)
    update_thread.start()
    print("🔄 Background update daemon started (will run every 30 minutes)")


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
    import os
    import datetime

    check_and_reload_model()

    text_hash = article_hash_override if article_hash_override else hashlib.md5(text.encode('utf-8')).hexdigest()
    processed = preprocessor.preprocess_for_model(text)

    # Initialize state dictionary to hold intermediate values
    state = {
        'text': text,
        'processed': processed,
        'model': model,
        'preprocessor': preprocessor,
        'clickbait_detector': clickbait_detector,
        'ai_detector': ai_detector,
        'claim_verifier': claim_verifier,
        'source_engine': source_engine,
        'url': url,
        'text_hash': text_hash,
        'prediction': 'REAL',  # will be updated
        'raw_confidence': 0.55,
        'ml_score': 0.5,
        'probs': [0.5, 0.5],
        'classes': ['FAKE', 'REAL'],
        'bert_triggered': False,
        'bert_result': None,
        'transformer_used': False,
        'transformer_model': "N/A",
        'transformer_score': None,
        'indicators': {},
        'sensationalism': 0.0,
        'cred_signal': 0.0,
        'nlp_nudge': 0.0,
        'nlp_score': 0.0,
        'clickbait_res': {},
        'clickbait_score': 0.0,
        'ai_res': {},
        'ai_score': 0.0,
        'source_trust': 50.0,
        'source_profile': None,
        'has_valid_url': False,
        'verification_res': {},
        'verification_results': [],
        'verification_status': '',
        'factcheck_score': 0.5,
        'evidence_count': 0,
        'evidence_quality': 0.0,
        'agreement_ratio': 0.5,
        'temporal_analysis': {"is_consistent": True, "risk_score": 0.0, "mismatches": []},
        'satire_res': {},
        'satire_score': 0.0,
        'is_satire': False,
        'stance_result': {},
        'article_stance': "NEUTRAL",
        'stance_confidence': 0.5,
        'is_factcheck_article': False,
        'matched_themes': [],
        'misinfo_multiplier': 1.0,
        'w_ml': 0.0,
        'w_source': 0.0,
        'w_factcheck': 0.0,
        'w_clickbait': 0.0,
        'w_nlp': 0.0,
        'w_ai': 0.0,
        'w_satire': 0.0,
        'w_sentiment': 0.0,
        'has_source_signal': False,
        'has_factcheck_signal': False,
        'credibility': 0.0,
        'reliability': 0.0,
        'is_sufficient': True,
        'category': '',
        'final_prediction': '',
        'confidence': 0.0,
        'is_clickbait': False,
        'is_ai_generated': False,
        'positive_factors': [],
        'negative_factors': [],
        'temporal_penalty': 0.0,
        'stale_years_found': [],
        'evidence_report': None,
    }

    # Helper functions to update state
    def _initialize_model_prediction():
        try:
            prediction = model.predict([state['processed']])[0]
            probs = model.predict_proba([state['processed']])[0]
            classes = list(model.classes_)
            raw_confidence = float(probs[classes.index(prediction)])
            # ml_score = P(REAL) directly from the model — this is the true signal
            real_idx = classes.index('REAL') if 'REAL' in classes else 1
            ml_score = float(probs[real_idx])
            state['prediction'] = prediction
            state['raw_confidence'] = raw_confidence
            state['probs'] = probs
            state['classes'] = classes
            state['ml_score'] = ml_score
        except Exception:
            state['prediction'] = 'REAL'
            state['raw_confidence'] = 0.55
            state['probs'] = [0.5, 0.5]
            state['classes'] = ['FAKE', 'REAL']
            state['ml_score'] = 0.5

    def _apply_transformer_classification():
        try:
            from utils.transformer_predictor import TransformerPredictor
            _transformer_predictor = TransformerPredictor()
            if _transformer_predictor.is_available:
                _lang = "en"
                try:
                    from utils.multilingual import MultilingualProcessor
                    _mp = MultilingualProcessor()
                    _det = _mp.detect_language(state['text'])
                    _lang = _det.get("language_code", "en")
                except Exception:
                    pass
                t_res = _transformer_predictor.predict(state['text'], language_code=_lang)
                if t_res:
                    state['transformer_used'] = True
                    state['bert_triggered'] = True
                    state['transformer_model'] = t_res.get("model_used", "transformer")
                    state['bert_result'] = t_res
                    state['transformer_score'] = t_res["probabilities"]["REAL"]
                    # Blend with ML score: 35% model, 65% transformer
                    state['ml_score'] = (state['ml_score'] * 0.35 + state['transformer_score'] * 0.65)
                    if state['ml_score'] >= 0.5:
                        state['prediction'] = 'REAL'
                        state['raw_confidence'] = state['ml_score']
                    else:
                        state['prediction'] = 'FAKE'
                        state['raw_confidence'] = 1.0 - state['ml_score']
        except Exception:
            pass

    def _apply_short_text_penalty():
        word_count = len(state['text'].split())
        if word_count < 150:
            length_factor = max(0.5, word_count / 150.0)
            state['raw_confidence'] *= length_factor
            # Shrink ml_score toward 0.5 for short text — model is unreliable on snippets
            state['ml_score'] = 0.5 + (state['ml_score'] - 0.5) * length_factor

    def _compute_nlp_indicators():
        state['indicators'] = state['preprocessor'].analyze_suspicious_indicators(state['text'])
        state['sensationalism'] = state['indicators'].get('sensationalism_score', 0.0)
        state['cred_signal'] = state['indicators'].get('credibility_score', 0.0)
        state['nlp_nudge'] = (state['cred_signal'] - state['sensationalism']) * 0.5
        state['nlp_score'] = max(0.0, min(1.0, 0.5 + state['nlp_nudge']))

    def _apply_clickbait_detection():
        if state['clickbait_detector'] is None:
            from utils.clickbait_detector import ClickbaitDetector
            state['clickbait_detector'] = ClickbaitDetector()
        title = state['url'] if state['url'] else state['text'][:100]
        state['clickbait_res'] = state['clickbait_detector'].detect(state['text'], title=title)
        state['clickbait_score'] = float(state['clickbait_res'].get("clickbait_score", 0.0))

    def _apply_ai_detection():
        if state['ai_detector'] is None:
            from utils.ai_detector import AIContentDetector
            state['ai_detector'] = AIContentDetector()
        state['ai_res'] = state['ai_detector'].detect(state['text'])
        state['ai_score'] = float(state['ai_res'].get("ai_score", 0.0))

    def _apply_source_trust():
        # Source Trust: only use actual URLs, never parse article text as domain
        state['has_valid_url'] = bool(state['url'] and state['url'].strip() and ('.' in state['url']) and len(state['url'].strip()) < 500)
        if state['has_valid_url']:
            domain_to_check = state['url']
            try:
                db_path_src = os.path.join(PROJECT_ROOT, "data", "truthshield.db")
                if os.path.exists(db_path_src):
                    conn = sqlite3.connect(db_path_src)
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    domain = state['source_engine'].clean_domain(domain_to_check)
                    cursor.execute("SELECT * FROM source_reputation WHERE domain = ?", (domain,))
                    row = cursor.fetchone()
                    conn.close()

                    if row:
                        state['source_trust'] = float(row["trust_score"])
                        state['source_profile'] = {
                            "domain": row["domain"],
                            "score": state['source_trust'],
                            "category": row["category"],
                            "bias": row["bias"],
                            "description": row["description"]
                        }
            except Exception:
                pass

            if not state['source_profile']:
                state['source_profile'] = state['source_engine'].get_trust_profile(domain_to_check)
                state['source_trust'] = float(state['source_profile'].get("score", 50.0))
        else:
            # No URL provided — use neutral defaults, do NOT parse article text as a domain
            state['source_profile'] = {
                "domain": "(no URL provided)",
                "score": 50.0,
                "category": "Unknown",
                "bias": "Unknown",
                "description": "No source URL was provided for domain reputation analysis."
            }
            state['source_trust'] = 50.0

    def _apply_claim_verification():
        if state['claim_verifier'] is None:
            from utils.claim_verifier import ClaimVerifier
            state['claim_verifier'] = ClaimVerifier()
        try:
            state['verification_res'] = state['claim_verifier'].verify_article(state['text'])
            state['verification_results'] = state['verification_res'].get("verification_results", [])
            state['verification_status'] = state['verification_res'].get("summary", "Unverified claims.")
            state['factcheck_score'] = float(state['verification_res'].get("overall_verification_score", 0.5))
            state['evidence_count'] = int(state['verification_res'].get("evidence_count", 0))
            state['evidence_quality'] = float(state['verification_res'].get("evidence_quality", 0.0))
            state['agreement_ratio'] = float(state['verification_res'].get("agreement_ratio", 0.5))
            state['temporal_analysis'] = state['verification_res'].get("temporal_analysis", {"is_consistent": True, "risk_score": 0.0, "mismatches": []})
        except Exception:
            state['verification_res'] = {}
            state['verification_results'] = []
            state['verification_status'] = "Fact verification unavailable."
            state['factcheck_score'] = 0.5
            state['evidence_count'] = 0
            state['evidence_quality'] = 0.0
            state['agreement_ratio'] = 0.5
            state['temporal_analysis'] = {"is_consistent": True, "risk_score": 0.0, "mismatches": []}

    def _apply_satire_detection():
        try:
            from utils.satire_detector import SatireDetector
            satire_detector = SatireDetector()
            state['satire_res'] = satire_detector.detect(state['text'], url=state['url'])
            state['satire_score'] = state['satire_res'].get("satire_score", 0.0)
            state['is_satire'] = state['satire_res'].get("is_satire", False)
        except Exception:
            state['satire_res'] = {"satire_score": 0.0, "is_satire": False}
            state['satire_score'] = 0.0
            state['is_satire'] = False

    def _apply_stance_detection():
        try:
            from utils.stance_detector import StanceDetector
            local_stance_detector = StanceDetector()
            stance_claims = state['verification_res'].get("claims", [])
        except Exception:
            local_stance_detector = None
            stance_claims = []

        if local_stance_detector:
            state['stance_result'] = local_stance_detector.detect(state['text'], claims=stance_claims,
                                                                   verification_results=state['verification_results'])
        else:
            state['stance_result'] = {
                "stance": "NEUTRAL",
                "stance_confidence": 0.5,
                "refutation_signals": [],
                "support_signals": [],
                "factchecker_references": [],
                "attribution_count": 0,
                "is_factcheck_article": False,
                "scores": {"refutation": 0.0, "support": 0.0}
            }
        state['article_stance'] = state['stance_result'].get("stance", "NEUTRAL")
        state['stance_confidence'] = state['stance_result'].get("stance_confidence", 0.5)
        state['is_factcheck_article'] = state['stance_result'].get("is_factcheck_article", False)
        # Only treat the article as REFUTES if it is verified as a factcheck article.
        if state['article_stance'] == "REFUTES" and not state['is_factcheck_article']:
            state['article_stance'] = "NEUTRAL"
            state['stance_confidence'] = 0.5

    def _apply_misinformation_pattern_matching():
        state['matched_themes'] = []
        state['misinfo_multiplier'] = 1.0
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
                    match = re.search(pattern, state['text'], re.IGNORECASE)
                    if match:
                        state['matched_themes'].append({
                            "theme": theme,
                            "match": match.group(0),
                            "multiplier": multiplier
                        })
                        # If the article is REFUTING the misinformation, do NOT penalize it
                        if state['article_stance'] != "REFUTES":
                            state['misinfo_multiplier'] *= (1.0 - multiplier)
        except Exception:
            pass

    def _compute_weighted_ensemble():
        state['has_source_signal'] = state['has_valid_url'] and state['source_profile'] is not None and state['source_profile'].get("category") not in ["Unknown", "Unverified Source"]
        state['has_factcheck_signal'] = state['evidence_count'] > 0

        if state['transformer_used']:
            state['w_ml'] = 0.40
            state['w_source'] = 0.15
            state['w_factcheck'] = 0.20
            state['w_clickbait'] = 0.05
            state['w_nlp'] = 0.05
            state['w_ai'] = 0.05
            state['w_satire'] = 0.05
            state['w_sentiment'] = 0.05
        else:
            state['w_ml'] = 0.31
            state['w_source'] = 0.20
            state['w_factcheck'] = 0.25
            state['w_clickbait'] = 0.05
            state['w_nlp'] = 0.06
            state['w_ai'] = 0.05
            state['w_satire'] = 0.08
            state['w_sentiment'] = 0.0

        if not state['has_source_signal']:
            state['w_ml'] += state['w_source'] * 0.5
            state['w_nlp'] += state['w_source'] * 0.5
            state['w_source'] = 0.0
        if not state['has_factcheck_signal']:
            state['w_ml'] += state['w_factcheck'] * 0.7
            state['w_nlp'] += state['w_factcheck'] * 0.3
            state['w_factcheck'] = 0.0
        if not state['has_source_signal'] and not state['has_factcheck_signal']:
            state['w_nlp'] += 0.05
            state['w_ml'] -= 0.05

        sentiment_bias = 1.0 - state['indicators'].get("sensationalism_score", 0.0)

        state['credibility'] = (
            state['ml_score'] * state['w_ml'] +
            (state['source_trust'] / 100.0) * state['w_source'] +
            state['factcheck_score'] * state['w_factcheck'] +
            (1.0 - state['clickbait_score']) * state['w_clickbait'] +
            state['nlp_score'] * state['w_nlp'] +
            (1.0 - state['ai_score']) * state['w_ai'] +
            (1.0 - state['satire_score']) * state['w_satire'] +
            sentiment_bias * state['w_sentiment']
        )

    def _apply_ml_confidence_anchoring():
        if not state['has_source_signal'] and not state['has_factcheck_signal']:
            if state['ml_score'] < 0.42:
                # ML is fairly confident this is FAKE — cap credibility
                state['ml_anchor_cap'] = 0.35 + state['ml_score'] * 0.35  # range: 0.35 to 0.497
                state['credibility'] = min(state['credibility'], state['ml_anchor_cap'])
            elif state['ml_score'] < 0.48:
                # ML is leaning FAKE but not very confident — soft cap
                state['ml_anchor_cap'] = 0.42 + state['ml_score'] * 0.20  # range: 0.42 to 0.516
                state['credibility'] = min(state['credibility'], state['ml_anchor_cap'])

    def _apply_stance_aware_adjustment():
        if state['article_stance'] == "REFUTES" and state['stance_confidence'] >= 0.2:
            # The article is debunking/fact-checking a false claim.
            # The ML model may have said FAKE because it sees misinformation keywords
            # (e.g., "hoax", "conspiracy", "vaccines cause autism") but those words
            # appear in the context of refutation, not endorsement.

            # 1. Dampen ML penalty: if ML said FAKE, reduce its negative pull
            if state['prediction'] == 'FAKE':
                # Flip the ML contribution: a fact-check article containing "fake" keywords is credible
                state['ml_boost'] = (1.0 - state['ml_score']) * state['stance_confidence'] * 0.35
                state['credibility'] += state['ml_boost']

            # 2. Boost from refutation evidence
            state['refutation_boost'] = state['stance_confidence'] * 0.15
            state['credibility'] += state['refutation_boost']

            # 3. Boost from fact-checker references
            if state['stance_result'].get("factchecker_references"):
                state['credibility'] += min(len(state['stance_result']["factchecker_references"]) * 0.05, 0.15)

            # 4. Boost from credibility indicators (experts, studies, etc.)
            state['cred_indicator_boost'] = state['cred_signal'] * 0.1
            state['credibility'] += state['cred_indicator_boost']
        elif state['article_stance'] == "SUPPORTS" and state['stance_confidence'] >= 0.3:
            # Article is promoting/endorsing a suspicious claim — apply extra penalty
            state['credibility'] -= state['stance_confidence'] * 0.1

    def _apply_redflag_penalty():
        redflag_count = state['indicators'].get('redflag_count', 0)
        if redflag_count > 0 and state['article_stance'] != "REFUTES":
            # Each red flag directly penalizes credibility (up to -0.55 total)
            state['redflag_penalty'] = min(redflag_count * 0.10, 0.55)
            state['credibility'] -= state['redflag_penalty']

            # Enforce ceiling overrides for multiple fabrication indicators
            if redflag_count >= 5:
                state['credibility'] = min(state['credibility'], 0.18)
            elif redflag_count >= 3:
                state['credibility'] = min(state['credibility'], 0.42)

    def _apply_misinformation_multiplier():
        state['credibility'] = state['credibility'] * state['misinfo_multiplier']
        state['credibility'] = float(max(0.0, min(1.0, state['credibility'])))

    def _apply_temporal_staleness_penalty():
        current_year = datetime.datetime.now().year
        stale_years_found = []
        freshness_words = ["latest", "new", "breaking", "just released", "current", "record", "today", "this week"]

        # Find all 4-digit years in text
        all_year_matches = list(re.finditer(r'\b(19\d{2}|20\d{2})\b', state['text']))

        # Find all freshness word indices
        freshness_indices = []
        text_lower = state['text'].lower()
        for word in freshness_words:
            for m in re.finditer(r'\b' + re.escape(word) + r'\b', text_lower):
                friendliness_indices.append(m.start())

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

        state['temporal_penalty'] = (0.20 + min((stale_count - 1) * 0.10, 0.20)) if stale_count > 0 else 0.0
        if state['temporal_analysis'].get("risk_score", 0.0) > 0.3:
            state['temporal_penalty'] = max(state['temporal_penalty'], 0.35)

        state['credibility'] -= state['temporal_penalty']
        state['credibility'] = float(max(0.0, min(1.0, state['credibility'])))

    def _check_evidence_sufficiency():
        state['is_sufficient'] = True
        source_is_known = state['source_profile'].get("category") not in ["Unverified Source", "Unknown"]
        if state['source_trust'] <= 55.0 and state['evidence_count'] == 0 and state['article_stance'] != "REFUTES":
            state['is_sufficient'] = False
        elif state['evidence_count'] > 0 and state['evidence_quality'] < 0.3:
            state['is_sufficient'] = False

    def _compute_reliability_score():
        source_is_known = state['source_profile'].get("category") not in ["Unverified Source", "Unknown"]
        source_rel_weight = 1.0 if source_is_known else 0.5
        ev_rel = min(state['evidence_count'] / 4.0, 1.0) if state['evidence_count'] > 0 else 0.2
        agree_rel = (abs(state['agreement_ratio'] - 0.5) * 2.0) if state['evidence_count'] > 0 else 0.5
        ml_rel = state['raw_confidence']
        # Stance detection adds to reliability when confident
        stance_rel = state['stance_confidence'] if state['article_stance'] in ("REFUTES", "SUPPORTS") else 0.3

        state['reliability'] = (
            source_rel_weight * 0.25 +
            ev_rel * 0.25 +
            agree_rel * 0.15 +
            ml_rel * 0.15 +
            stance_rel * 0.20
        )
        state['reliability'] = float(max(0.0, min(1.0, state['reliability'])))

    def _determine_5_level_verdict():
        if state['credibility'] >= 0.85:
            state['category'] = "Highly Credible"
        elif state['credibility'] >= 0.65:
            state['category'] = "Likely Real"
        elif state['credibility'] >= 0.45:
            state['category'] = "Uncertain"
        elif state['credibility'] >= 0.20:
            state['category'] = "Likely Fake"
        else:
            state['category'] = "High Risk Misinformation"

        # Override: fact-check articles with strong refutation should not be "Uncertain" or worse
        if state['is_factcheck_article'] and state['credibility'] >= 0.40 and state['category'] in ("Uncertain", "Likely Fake", "High Risk Misinformation"):
            state['category'] = "Likely Real"
            state['credibility'] = max(state['credibility'], 0.65)

        # Apply evidence sufficiency fallback: if there is no evidence and source is untrusted,
        # cap credibility and verdict at 'Uncertain' to prevent false-positives on unverified/AI articles.
        if not state['is_sufficient'] and not state['is_factcheck_article']:
            if state['credibility'] > 0.58:
                state['credibility'] = 0.58
            if state['credibility'] >= 0.45:
                state['category'] = "Uncertain"
                state['reliability'] = min(state['reliability'], 0.40)

    def _compute_final_confidence():
        evidence_confidence = min(state['evidence_count'] / 3.0, 1.0) if state['evidence_count'] > 0 else 0.3
        state['confidence'] = (
            state['reliability'] * 0.5 +
            evidence_confidence * 0.25 +
            state['stance_confidence'] * 0.25
        )
        state['confidence'] = float(min(max(state['confidence'], 0.05), 0.99))

    def _determine_final_prediction():
        if not state['has_source_signal'] and not state['has_factcheck_signal']:
            base_threshold = 0.55
            # lower threshold when NLP signals strongly suggest fake
            if state['nlp_score'] < 0.4:
                threshold = base_threshold - 0.05  # 0.50, easier to call FAKE
            elif state['indicators'].get('redflag_count', 0) >= 2:
                threshold = base_threshold - 0.08  # 0.47
            else:
                threshold = base_threshold
        else:
            threshold = 0.50

        state['final_prediction'] = "REAL" if state['credibility'] >= threshold else "FAKE"

    def _set_subclassification_flags():
        state['is_clickbait'] = state['clickbait_score'] > 0.6
        state['is_ai_generated'] = state['ai_score'] > 0.75
        # Note: is_satire already set in _apply_satire_detection, but we also have a check for source_profile category
        if state['source_profile'] and state['source_profile'].get("category") in ["Satire / Parody", "Parody"]:
            state['is_satire'] = True
        # Satire override
        if state['is_satire'] or state['satire_score'] > 0.42:
            state['is_satire'] = True
            state['final_prediction'] = "FAKE"
            state['category'] = "Satire / Parody"
            state['credibility'] = min(state['credibility'], 0.35)

    def _compute_risk_factor_breakdown():
        # Positive/Negative factors
        state['positive_factors'] = []
        state['negative_factors'] = []

        # Stance-based factors
        if state['article_stance'] == "REFUTES" and state['stance_confidence'] >= 0.2:
            refutation_sigs = ", ".join(state['stance_result']["refutation_signals"][:3]) if state['stance_result']["refutation_signals"] else "refutation language detected"
            state['positive_factors'].append({"factor": "Fact-Check / Debunking Article", "detail": f"Article refutes claims with evidence ({refutation_sigs}).", "impact": f"+{int(state['stance_confidence'] * 30)}%"})
            if state['stance_result'].get("factchecker_references"):
                refs = ", ".join(state['stance_result']["factchecker_references"][:3])
                state['positive_factors'].append({"factor": "Fact-Checker References", "detail": f"Cites fact-checking sources: {refs}.", "impact": "+15%"})
        elif state['article_stance'] == "SUPPORTS" and state['stance_confidence'] >= 0.3:
            support_sigs = ", ".join(state['stance_result']["support_signals"][:3]) if state['stance_result']["support_signals"] else "promotional language"
            state['negative_factors'].append({"factor": "Promotes Unverified Claims", "detail": f"Article endorses claims without evidence ({support_sigs}).", "impact": f"-{int(state['stance_confidence'] * 20)}%"})

        if state['source_trust'] >= 75.0:
            state['positive_factors'].append({"factor": "Trusted Publisher", "detail": f"Source {state['source_profile']['domain']} has high trust ({state['source_trust']}%).", "impact": "+20%"})
        elif state['source_trust'] <= 40.0:
            state['negative_factors'].append({"factor": "Untrusted Source", "detail": f"Source {state['source_profile']['domain']} is flagged as low-trust ({state['source_trust']}%).", "impact": "-20%"})

        for theme_match in state['matched_themes']:
            # If article refutes the misinformation, show it as informational, not a penalty
            if state['article_stance'] == "REFUTES":
                state['positive_factors'].append({
                    "factor": f"Addresses: {theme_match['theme']}",
                    "detail": f"Article discusses and debunks {theme_match['theme'].lower()} topic (\"{theme_match['match']}\").",
                    "impact": "+5%"
                })
            else:
                state['negative_factors'].append({
                    "factor": f"Misinformation: {theme_match['theme']}",
                    "detail": f"Content matches known pattern for {theme_match['theme'].lower()} (\"{theme_match['match']}\").",
                    "impact": f"-{int(theme_match['multiplier'] * 100)}%"
                })

        # Red-flag pattern matches
        redflags_list = state['indicators'].get('redflags', [])
        if redflags_list and state['article_stance'] != "REFUTES":
            rf_examples = ", ".join(f'"{rf}"' for rf in redflags_list[:3])
            state['negative_factors'].append({
                "factor": "Fabrication Indicators",
                "detail": f"Detected {len(redflags_list)} red-flag phrases: {rf_examples}.",
                "impact": f"-{min(len(redflags_list) * 6, 30)}%"
            })

        if state['evidence_count'] > 0:
            if state['agreement_ratio'] >= 0.75:
                state['positive_factors'].append({"factor": "Fact Matches", "detail": f"High RAG evidence agreement ({state['agreement_ratio'] * 100:.0f}% supporting).", "impact": "+25%"})
            elif state['agreement_ratio'] <= 0.25:
                state['negative_factors'].append({"factor": "Contradicted Claims", "detail": f"RAG checks find direct contradictions in claims.", "impact": "-25%"})

        if not state['temporal_analysis'].get("is_consistent", True):
            state['negative_factors'].append({"factor": "Temporal Drift", "detail": "Reused old statistics or outdated event timeline detected.", "impact": f"-{int(state['temporal_analysis']['risk_score'] * 30)}%"})

        if state['is_clickbait']:
            state['negative_factors'].append({"factor": "Clickbait Headlines", "detail": f"Sensationalized framing detected ({state['clickbait_score'] * 100:.0f}% clickbait).", "impact": "-5%"})
        else:
            state['positive_factors'].append({"factor": "Editorial Title", "detail": "Neutral, objective title formatting.", "impact": "+5%"})

        if state['is_ai_generated']:
            state['negative_factors'].append({"factor": "Linguistic Uniformity", "detail": f"High similarity to AI-generated text ({state['ai_score'] * 100:.0f}% AI score).", "impact": "-5%"})

        if state['raw_confidence'] > 0.75 and state['prediction'] == 'REAL':
            state['positive_factors'].append({"factor": "Model Signal", "detail": f"Classifier strongly validates text patterns.", "impact": "+35%"})
        elif state['raw_confidence'] > 0.75 and state['prediction'] == 'FAKE' and state['article_stance'] != "REFUTES":
            state['negative_factors'].append({"factor": "Stylistic Flags", "detail": f"Classifier detects typical misinformation patterns.", "impact": "-35%"})
        elif state['raw_confidence'] > 0.75 and state['prediction'] == 'FAKE' and state['article_stance'] == "REFUTES":
            state['positive_factors'].append({"factor": "Misinformation Keywords (Debunking Context)", "detail": "Classifier detected misinformation-related keywords, but article uses them in a refutation/fact-check context.", "impact": "+10%"})

    def _update_audit_log():
        try:
            db_path = os.path.join(PROJECT_ROOT, "data", "truthshield.db")
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()

                # Calculate dynamic distributions
                cursor.execute("SELECT COUNT(*) FROM feedback")
                feedback_count = cursor.fetchone()[0]

                monthly_accuracy = 0.90  # default
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
                verdict_count_for_category = verdict_counts.get(state['category'], 0) + 1
                verdict_counts[state['category']] = verdict_count_for_category

                # Get source distribution (trust score categories)
                cursor.execute("""
                    SELECT CAST(source_score AS INTEGER), COUNT(*)
                    FROM prediction_audit
                    WHERE timestamp >= datetime('now', '-30 days')
                    GROUP BY CAST(source_score AS INTEGER)
                """)
                source_dist = {str(k): v for k, v in cursor.fetchall()}
                source_trust_int = int(state['source_trust'])
                source_dist[str(source_trust_int)] = source_dist.get(str(source_trust_int), 0) + 1

                feat_contrib = {
                    "ml_model": round(state['w_ml'], 4),
                    "source_trust": round(state['w_source'], 4),
                    "fact_check": round(state['w_factcheck'], 4),
                    "nlp_indicators": round(state['w_nlp'], 4),
                    "clickbait": round(state['w_clickbait'], 4),
                    "ai_generated": round(state['w_ai'], 4),
                    "satire": round(state['w_satire'], 4),
                }
                cursor.execute("""
                    INSERT INTO prediction_audit (
                        article_hash, verdict, final_prediction, confidence, credibility, reliability,
                        source_score, factcheck_score, clickbait_score, ai_score,
                        features_contribution, monthly_accuracy, source_distribution, prediction_distribution
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    state['text_hash'], state['category'], state['final_prediction'], state['confidence'], state['credibility'], state['reliability'],
                    state['source_trust'], state['factcheck_score'], state['clickbait_score'], state['ai_score'],
                    json.dumps(feat_contrib), monthly_accuracy, json.dumps(source_dist), json.dumps(verdict_counts)
                ))
                conn.commit()
                conn.close()
        except Exception:
            pass

    def _map_to_display_verdict():
        if state['credibility'] >= 0.65:
            state['display_verdict'] = "\U0001f7e2 Likely True"
        elif state['credibility'] >= 0.45:
            state['display_verdict'] = "\U0001f7e1 Needs Verification"
        else:
            state['display_verdict'] = "\U0001f534 Likely Misleading"

    def _generate_evidence_report():
        try:
            from utils.evidence_engine import EvidenceEngine
            _ee = EvidenceEngine()
            _partial = {
                'prediction': state['final_prediction'], 'confidence': state['confidence'],
                'credibility': state['credibility'], 'reliability': state['reliability'],
                'category': state['category'], 'clickbait_score': state['clickbait_score'],
                'ai_score': state['ai_score'], 'source_trust': state['source_trust'],
                'source_profile': state['source_profile'], 'verification_results': state['verification_results'],
                'indicators': state['indicators'], 'bert_triggered': state['bert_triggered'],
                'is_clickbait': state['is_clickbait'], 'is_ai_generated': state['is_ai_generated'],
                'is_satire': state['is_satire'], 'nlp_score': state['nlp_score'],
                'factcheck_score': state['factcheck_score'], 'article_stance': state['article_stance'],
                'stance_confidence': state['stance_confidence'], 'ml_score': state['ml_score'],
                'evidence_count': state['evidence_count'], 'matched_themes': state['matched_themes'],
                'satire_score': state['satire_score'], 'temporal_analysis': state['temporal_analysis'],
                'stance': state['stance_result'], 'transformer_used': state['transformer_used'],
                'transformer_model': state['transformer_model'],
            }
            state['evidence_report'] = _ee.generate_report(_partial)
        except Exception:
            state['evidence_report'] = None

    # Execute the pipeline in order
    _initialize_model_prediction()
    _apply_transformer_classification()
    _apply_short_text_penalty()
    _compute_nlp_indicators()
    _apply_clickbait_detection()
    _apply_ai_detection()
    _apply_source_trust()
    _apply_claim_verification()
    _apply_satire_detection()
    _apply_stance_detection()
    _apply_misinformation_pattern_matching()
    _compute_weighted_ensemble()
    _apply_ml_confidence_anchoring()
    _apply_stance_aware_adjustment()
    _apply_redflag_penalty()
    _apply_misinformation_multiplier()
    _apply_temporal_staleness_penalty()
    _check_evidence_sufficiency()
    _compute_reliability_score()
    _determine_5_level_verdict()
    _compute_final_confidence()
    _determine_final_prediction()
    _set_subclassification_flags()
    _compute_risk_factor_breakdown()
    _update_audit_log()
    _map_to_display_verdict()
    _generate_evidence_report()

    # Return the final result dictionary
    return {
        'prediction': state['final_prediction'],
        'article_hash': state['text_hash'],
        'confidence': state['confidence'],
        'credibility': state['credibility'],
        'reliability': state['reliability'],
        'category': state['category'],
        'display_verdict': state['display_verdict'],
        'clickbait_score': state['clickbait_score'],
        'ai_score': state['ai_score'],
        'source_trust': state['source_trust'],
        'source_profile': state['source_profile'],
        'verification_results': state['verification_results'],
        'verification_status': state['verification_status'],
        'indicators': state['indicators'],
        'bert_triggered': state['bert_triggered'],
        'bert_result': state['bert_result'],
        'transformer_used': state['transformer_used'],
        'transformer_model': state['transformer_model'],
        'is_sufficient': state['is_sufficient'],
        'is_clickbait': state['is_clickbait'],
        'is_ai_generated': state['is_ai_generated'],
        'is_satire': state['is_satire'],
        'positive_factors': state['positive_factors'],
        'negative_factors': state['negative_factors'],
        'temporal_analysis': state['temporal_analysis'],
        'stance': state['stance_result'],
        'ml_score': state['ml_score'],
        'nlp_score': state['nlp_score'],
        'factcheck_score': state['factcheck_score'],
        'article_stance': state['article_stance'],
        'stance_confidence': state['stance_confidence'],
        'evidence_count': state['evidence_count'],
        'matched_themes': state['matched_themes'],
        'satire_score': state['satire_score'],
        'evidence_report': state['evidence_report'],
    }

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
    result = predict_article(
        eng_text, model, preprocessor,
        clickbait_detector, ai_detector,
        claim_verifier, source_engine,
        url=request.url,
        article_hash_override=article_hash
    )

    # 3. Package response
    return PredictResponse(
        prediction=result["prediction"],
        article_hash=article_hash,
        confidence=result["confidence"],
        credibility=result["credibility"],
        reliability=result["reliability"],
        summary=result["summary"],
        is_fake=(result["prediction"] == "FAKE")
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Full analysis endpoint returning all available signals."""
    import hashlib
    text = request.text
    article_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

    # Translate multilingual content
    res_lang = multilingual_processor.process_for_analysis(text)  # type: ignore[union-attr]
    eng_text = res_lang["processed_text"]

    # Get full analysis
    result = predict_article(
        eng_text, model, preprocessor,
        clickbait_detector, ai_detector,
        claim_verifier, source_engine,
        url=request.url,
        article_hash_override=article_hash
    )

    # Package response
    return AnalyzeResponse(
        prediction=result["prediction"],
        article_hash=article_hash,
        confidence=result["confidence"],
        credibility=result["credibility"],
        reliability=result["reliability"],
        category=result["category"],
        clickbait_score=result["clickbait_score"],
        ai_score=result["ai_score"],
        source_trust=result["source_trust"],
        verification_results=result["verification_results"],
        summary=result["summary"],
        language=result["indicators"].get("language", "en"),
        was_translated=result["indicators"].get("was_translated", False)
    )


@app.get("/history")
async def get_history(limit: int = 50, offset: int = 0):
    """Get paginated analysis history."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, user_email, title, prediction, confidence, credibility,
                   reliability, category, clickbait_score, ai_score, source_trust,
                   verification_status, timestamp
            FROM history
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        """, (limit, offset))

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
                "reliability": r["reliability"],
                "category": r["category"] if r["category"] else r["prediction"],
                "clickbait_score": r["clickbait_score"],
                "ai_score": r["ai_score"],
                "source_trust": r["source_trust"],
                "verification_status": r["verification_status"],
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

            # Daily aggregation
            if pred_date not in daily:
                daily[pred_date] = {"predictions": 0, "feedback": 0, "agree": 0}
            daily[pred_date]["feedback"] += 1
            if agreement:
                daily[pred_date]["agree"] += 1

            # Verdict breakdown
            if final_pred in verdict_breakdown:
                if agreement:
                    verdict_breakdown[final_pred]["agree"] += 1
                else:
                    verdict_breakdown[final_pred]["disagree"] += 1

        # Get daily prediction counts
        cursor.execute("""
            SELECT date(timestamp) as pred_date, COUNT(*) as pred_count
            FROM prediction_audit
            WHERE timestamp >= datetime('now', ?)
            GROUP BY date(timestamp)
        """, (f"-{days} days",))
        pred_counts = dict(cursor.fetchall())
        for date_str, count in pred_counts.items():
            if date_str in daily:
                daily[date_str]["predictions"] = count

        # Calculate daily accuracy
        for date_str in daily:
            if daily[date_str]["feedback"] > 0:
                daily[date_str]["accuracy"] = daily[date_str]["agree"] / daily[date_str]["feedback"]
            else:
                daily[date_str]["accuracy"] = 0.0

        feedback_coverage = total_feedback / total_predictions if total_predictions > 0 else 0.0
        overall_accuracy = total_agree / total_feedback if total_feedback > 0 else 0.0

        return {
            "total_predictions": total_predictions,
            "feedback_coverage": feedback_coverage,
            "overall_accuracy": overall_accuracy,
            "daily": daily,
            "verdict_breakdown": verdict_breakdown
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api_server:app", host="0.0.0.0", port=8000, reload=False)