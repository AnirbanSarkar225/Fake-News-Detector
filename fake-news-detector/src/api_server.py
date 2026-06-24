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

def predict_article(text, model, preprocessor, clickbait_detector=None, ai_detector=None, claim_verifier=None, source_engine=None, url=None):
    """
    Advanced credibility analysis decision engine.
    """
    import hashlib
    import json
    import sqlite3
    
    check_and_reload_model()
    
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
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
        
    bert_triggered = False
    bert_result = None
    if 0.45 <= raw_confidence <= 0.75:
        try:
            from utils.bert_predictor import BertPredictor
            bert_predictor = BertPredictor()
            if bert_predictor.is_available:
                bert_res = bert_predictor.predict(text)
                if bert_res:
                    bert_triggered = True
                    bert_result = bert_res
                    b_pred = bert_res['prediction']
                    b_conf = bert_res['confidence']
                    bert_ml_score = (0.5 + b_conf * 0.5) if b_pred == 'REAL' else (0.5 - b_conf * 0.5)
                    ml_score = (ml_score + bert_ml_score) / 2.0
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
        verification_results = []
        verification_status = "Fact verification unavailable."
        factcheck_score = 0.5
        evidence_count = 0
        evidence_quality = 0.0
        agreement_ratio = 0.5
        temporal_analysis = {"is_consistent": True, "risk_score": 0.0, "mismatches": []}
        
    # ── Stance Detection ──
    try:
        stance_claims = verification_res.get("claims", [])
    except NameError:
        stance_claims = []
    stance_result = stance_detector.detect(text, claims=stance_claims,
                                           verification_results=verification_results)
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
    except Exception as e:
        print(f"WARNING: Misinfo patterns DB fetch failed: {e}")

    # ── 9. Weighted Ensemble Decision Engine (Stance-Aware) ──
    # Dynamic weight redistribution: when we have no real signal for source_trust
    # or factcheck, those weights go to the ML model (the only real signal).
    has_source_signal = has_valid_url and source_profile is not None and source_profile.get("category") not in ["Unknown", "Unverified Source"]
    has_factcheck_signal = evidence_count > 0
    
    w_ml = 0.35
    w_source = 0.20
    w_factcheck = 0.25
    w_clickbait = 0.05
    w_nlp = 0.10
    w_ai = 0.05
    
    # Redistribute phantom weights to ML when we have no real data
    if not has_source_signal:
        w_ml += w_source * 0.6       # 60% of source weight -> ML
        w_nlp += w_source * 0.4      # 40% of source weight -> NLP
        w_source = 0.0
    if not has_factcheck_signal:
        w_ml += w_factcheck * 0.7    # 70% of factcheck weight -> ML
        w_nlp += w_factcheck * 0.3   # 30% of factcheck weight -> NLP
        w_factcheck = 0.0
    
    credibility = (
        ml_score * w_ml +
        (source_trust / 100.0) * w_source +
        factcheck_score * w_factcheck +
        (1.0 - clickbait_score) * w_clickbait +
        nlp_score * w_nlp +
        (1.0 - ai_score) * w_ai
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
    
    # Apply misinformation patterns multiplier
    credibility = credibility * misinfo_multiplier
    credibility = float(max(0.0, min(1.0, credibility)))
    
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
    
    # Set final binary prediction (calibrated threshold if no external metadata is present)
    threshold = 0.55 if (not has_source_signal and not has_factcheck_signal) else 0.50
    final_prediction = "REAL" if credibility >= threshold else "FAKE"
    
    # Sub-classification flags
    is_clickbait = clickbait_score > 0.6
    is_ai_generated = ai_score > 0.75
    is_satire = source_profile.get("category") in ["Satire / Parody", "Parody"]
    
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
                "ml_model": 0.35,
                "source_trust": 0.20,
                "fact_check": 0.25,
                "nlp_indicators": 0.10,
                "clickbait": 0.05,
                "ai_generated": 0.05
            }
            cursor.execute("""
                INSERT INTO prediction_audit (
                    article_hash, verdict, confidence, credibility, reliability, 
                    source_score, factcheck_score, clickbait_score, ai_score, 
                    features_contribution, monthly_accuracy, source_distribution, prediction_distribution
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                text_hash, category, confidence, credibility, reliability,
                source_trust, factcheck_score, clickbait_score, ai_score,
                json.dumps(feat_contrib), monthly_accuracy, json.dumps(source_dist), json.dumps(verdict_counts)
            ))
            conn.commit()
            conn.close()
    except Exception:
        pass
        
    return {
        'prediction': final_prediction,
        'confidence': confidence,
        'credibility': credibility,
        'reliability': reliability,
        'category': category,
        'clickbait_score': clickbait_score,
        'ai_score': ai_score,
        'source_trust': source_trust,
        'source_profile': source_profile,
        'verification_results': verification_results,
        'verification_status': verification_status,
        'indicators': indicators,
        'bert_triggered': bert_triggered,
        'bert_result': bert_result,
        'is_sufficient': is_sufficient,
        'is_clickbait': is_clickbait,
        'is_ai_generated': is_ai_generated,
        'is_satire': is_satire,
        'positive_factors': positive_factors,
        'negative_factors': negative_factors,
        'temporal_analysis': temporal_analysis,
        'stance': stance_result
    }

# ------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------

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
    text = request.text
    
    # 1. Translate multilingual content
    res_lang = multilingual_processor.process_for_analysis(text)
    eng_text = res_lang["processed_text"]
    
    # 2. Get predictions using unified pipeline
    res = predict_article(
        eng_text, model, preprocessor, clickbait_detector, ai_detector,
        claim_verifier, source_engine, request.url
    )
    
    return PredictResponse(
        prediction=res['prediction'],
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
    text = request.text
    url = request.url or ""
    
    # 1. Multilingual processing
    res_lang = multilingual_processor.process_for_analysis(text)
    eng_text = res_lang["processed_text"]
    detected_lang_name = res_lang["language_name"]
    was_translated = res_lang["was_translated"]
    
    # 2. Run prediction pipeline
    res = predict_article(
        eng_text, model, preprocessor, clickbait_detector, ai_detector,
        claim_verifier, source_engine, url
    )
    
    return AnalyzeResponse(
        prediction=res['prediction'],
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
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedback (user_email, text, model_prediction, user_verdict, rating, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            request.user_email, request.text, request.model_prediction,
            request.user_verdict, request.rating, request.notes
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

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Initializing TruthShield API services...")
    load_services()
    
    port = 8000
    print(f"🌐 Starting production API server on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
