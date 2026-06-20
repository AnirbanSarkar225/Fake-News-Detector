"""
API Server for TruthShield Fake News Detector.

Provides REST endpoints for predictions, advanced analyses, user feedback,
and historical search. Integrates with the database, clickbait detector,
AI content detector, claim verifier, multilingual translator, and source trust engine.
"""

import os
import sys
import sqlite3
import joblib
import json
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Body
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
    global claim_verifier, multilingual_processor, source_engine, last_loaded_mtime
    
    # Initialize utility modules
    preprocessor = TextPreprocessor()
    clickbait_detector = ClickbaitDetector()
    ai_detector = AIContentDetector()
    claim_verifier = ClaimVerifier()
    multilingual_processor = MultilingualProcessor()
    source_engine = SourceEngine()
    
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
    except Exception:
        prediction = 'REAL'
        raw_confidence = 0.55
        probs = [0.5, 0.5]
        classes = ['FAKE', 'REAL']
        
    ml_score = 0.5 + (raw_confidence * 0.5) if prediction == 'REAL' else 0.5 - (raw_confidence * 0.5)
    
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
                    bert_score = 0.5 + (b_conf * 0.5) if b_pred == 'REAL' else 0.5 - (b_conf * 0.5)
                    ml_score = (ml_score + bert_score) / 2.0
                    if ml_score >= 0.5:
                        prediction = 'REAL'
                        raw_confidence = (ml_score - 0.5) * 2.0
                    else:
                        prediction = 'FAKE'
                        raw_confidence = (0.5 - ml_score) * 2.0
        except Exception:
            pass
            
    word_count = len(text.split())
    if word_count < 150:
        length_factor = max(0.3, word_count / 150.0)
        raw_confidence *= length_factor
        
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
    domain_to_check = url if url else text
    
    source_trust = 50.0
    source_profile = None
    try:
        db_path = os.path.join(PROJECT_ROOT, "data", "truthshield.db")
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
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
                    misinfo_multiplier *= (1.0 - multiplier)
    except Exception:
        pass

    # ── 9. Weighted Ensemble Decision Engine ──
    credibility = (
        ml_score * 0.35 +
        (source_trust / 100.0) * 0.20 +
        factcheck_score * 0.25 +
        (1.0 - clickbait_score) * 0.05 +
        nlp_score * 0.10 +
        (1.0 - ai_score) * 0.05
    )
    # Apply misinformation patterns multiplier
    credibility = credibility * misinfo_multiplier
    credibility = float(max(0.0, min(1.0, credibility)))
    
    # ── 10. Evidence Sufficiency Check ──
    is_sufficient = True
    source_is_known = source_profile.get("category") not in ["Unverified Source", "Unknown"]
    if source_trust <= 55.0 and evidence_count == 0:
        is_sufficient = False
    elif evidence_count > 0 and evidence_quality < 0.3:
        is_sufficient = False
        
    # ── 11. Reliability Score ──
    source_rel_weight = 1.0 if source_is_known else 0.5
    ev_rel = min(evidence_count / 4.0, 1.0) if evidence_count > 0 else 0.2
    agree_rel = (abs(agreement_ratio - 0.5) * 2.0) if evidence_count > 0 else 0.5
    ml_rel = raw_confidence
    
    reliability = (
        source_rel_weight * 0.3 +
        ev_rel * 0.3 +
        agree_rel * 0.2 +
        ml_rel * 0.2
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
        
    # Apply evidence sufficiency fallback
    if not is_sufficient:
        category = "Uncertain"
        reliability = min(reliability, 0.40)
        
    # Final adjusted confidence (calibrated and clipped)
    confidence = min(reliability, 0.99)
    
    # Set final binary prediction
    final_prediction = "REAL" if credibility >= 0.5 else "FAKE"
    
    # Sub-classification flags
    is_clickbait = clickbait_score > 0.6
    is_ai_generated = ai_score > 0.75
    is_satire = source_profile.get("category") in ["Satire / Parody", "Parody"]
    
    # Risk Factor Breakdown
    positive_factors = []
    negative_factors = []
    
    if source_trust >= 75.0:
        positive_factors.append({"factor": "Trusted Publisher", "detail": f"Source {source_profile['domain']} has high trust ({source_trust}%).", "impact": "+20%"})
    elif source_trust <= 40.0:
        negative_factors.append({"factor": "Untrusted Source", "detail": f"Source {source_profile['domain']} is flagged as low-trust ({source_trust}%).", "impact": "-20%"})
        
    for theme_match in matched_themes:
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
    elif raw_confidence > 0.75 and prediction == 'FAKE':
        negative_factors.append({"factor": "Stylistic Flags", "detail": f"Classifier detects typical misinformation patterns.", "impact": "-35%"})
        
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
        'temporal_analysis': temporal_analysis
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
