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
    
    # Base history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            title TEXT,
            text TEXT,
            prediction TEXT,
            confidence REAL,
            credibility REAL,
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
    
    # Run migration checks: Ensure newer columns are added to existing history table
    try:
        cursor.execute("PRAGMA table_info(history)")
        columns = [row[1] for row in cursor.fetchall()]
        
        migrations = {
            "category": "TEXT",
            "clickbait_score": "REAL",
            "ai_score": "REAL",
            "source_trust": "REAL",
            "verification_status": "TEXT"
        }
        
        for col, col_type in migrations.items():
            if col not in columns:
                cursor.execute(f"ALTER TABLE history ADD COLUMN {col} {col_type}")
                print(f"🔧 Database migration: added '{col}' to history table.")
    except Exception as e:
        print(f"⚠️ Database migration error: {e}")
        
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

def run_predictions_on_text(eng_text: str) -> tuple:
    """Predict label and confidence using ML model, falling back to heuristics."""
    check_and_reload_model()
    
    # Preprocess text
    clean_text = preprocessor.preprocess_for_model(eng_text)
    
    if not model:
        # Fallback heuristic prediction
        lower_text = eng_text.lower()
        suspicion_ratio = len(preprocessor.analyze_suspicious_indicators(eng_text).get("suspicious_patterns", []))
        if suspicion_ratio > 3 or "unbelievable" in lower_text or "secret report" in lower_text:
            return "FAKE", 0.60
        return "REAL", 0.65

    try:
        # Predict using ensemble pipeline
        prediction = model.predict([clean_text])[0]
        
        # Soft voting probabilities
        probs = model.predict_proba([clean_text])[0]
        classes = list(model.classes_)
        confidence = float(probs[classes.index(prediction)])
        
        return prediction, confidence
    except Exception as e:
        print(f"⚠️ Model prediction error: {e}")
        # Secondary fallback
        return "REAL", 0.50

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
    
    # 2. Get predictions
    pred, conf = run_predictions_on_text(eng_text)
    
    # Calculate credibility (0.0 to 1.0)
    if pred == "REAL":
        cred = 0.5 + (conf * 0.5)
    else:
        cred = 0.5 - (conf * 0.5)
        
    indicators = preprocessor.analyze_suspicious_indicators(eng_text)
    summary = "Verified text shows typical signs of factual reporting." if pred == "REAL" else f"Identified {len(indicators.get('suspicious_patterns', []))} suspicious indicators."
    
    # Save to history db as anonymous
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        title = text[:50] + "..." if len(text) > 50 else text
        cursor.execute(
            "INSERT INTO history (user_email, title, text, prediction, confidence, credibility) VALUES (?, ?, ?, ?, ?, ?)",
            ("anonymous@truthshield.ai", title, text, pred, conf, cred)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass # Ignore db insert failures for predict endpoint
        
    return PredictResponse(
        prediction=pred,
        confidence=conf,
        credibility=cred,
        summary=summary,
        is_fake=(pred == "FAKE")
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
    detected_lang = res_lang["language_code"]
    detected_lang_name = res_lang["language_name"]
    was_translated = res_lang["was_translated"]
    
    # 2. Base ML Model Predictions
    pred, conf = run_predictions_on_text(eng_text)
    
    # 3. Clickbait Detection
    title_text = text[:100] if not url else url
    clickbait_res = clickbait_detector.detect(text, title=title_text)
    clickbait_score = float(clickbait_res.get("clickbait_score", 0.0))
    
    # 4. AI-Generated Content Check
    ai_res = ai_detector.detect(eng_text)
    ai_score = float(ai_res.get("ai_score", 0.0))
    
    # 5. Domain reputation & credibility score
    domain_to_check = url if url else text
    source_trust = float(source_engine.get_source_credibility_score(domain_to_check))
    domain_profile = source_engine.get_trust_profile(domain_to_check)
    
    # 6. Deep Fact Verification (RAG)
    try:
        verification_res = claim_verifier.verify_article(eng_text)
        verification_results = verification_res.get("verification_results", [])
        verification_status = verification_res.get("summary", "Unverified claims.")
    except Exception as e:
        print(f"⚠️ Fact checking failed: {e}")
        verification_results = []
        verification_status = "Fact verification unavailable."

    # 7. Post-predict Multi-class refinement (Option B)
    category = pred
    if pred == "REAL":
        category = "REAL"
    else: # pred == "FAKE"
        if clickbait_score > 0.7:
            category = "CLICKBAIT"
        elif domain_profile.get("category") in ["Satire / Parody", "Parody"]:
            category = "SATIRE"
        elif conf < 0.65 or source_trust < 50.0:
            category = "MISLEADING"
        else:
            category = "FAKE"
            
    # Calculate credibility (0.0 to 1.0)
    # Scale based on source trust, AI and clickbait scores
    base_cred = 0.5 + (conf * 0.5) if pred == "REAL" else 0.5 - (conf * 0.5)
    # Fine tune with weights
    adjusted_cred = (base_cred * 0.5) + ((source_trust / 100.0) * 0.3) - (clickbait_score * 0.1) - (ai_score * 0.1)
    credibility = float(max(0.0, min(1.0, adjusted_cred)))
    
    # Generate summary message
    summary_parts = []
    if category == "REAL":
        summary_parts.append(f"Authentic article ({detected_lang_name}).")
    else:
        summary_parts.append(f"Classified as {category} news.")
        
    if source_trust > 80.0:
        summary_parts.append("Published by a highly trusted publisher.")
    elif source_trust < 40.0:
        summary_parts.append("Caution: Source domain has low trustworthiness.")
        
    if clickbait_score > 0.6:
        summary_parts.append("Sensationalized headlines detected.")
        
    if ai_score > 0.7:
        summary_parts.append("Linguistic patterns highly resemble AI-generated text.")
        
    summary = " ".join(summary_parts)
    
    # 8. Store results in Database History
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        title = text[:80] + "..." if len(text) > 80 else text
        cursor.execute("""
            INSERT INTO history (
                user_email, title, text, prediction, confidence, credibility, 
                category, clickbait_score, ai_score, source_trust, verification_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "anonymous@truthshield.ai", title, text, pred, conf, credibility,
            category, clickbait_score, ai_score, source_trust, verification_status
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Failed to save history to DB: {e}")
        
    return AnalyzeResponse(
        prediction=pred,
        confidence=conf,
        credibility=credibility,
        category=category,
        clickbait_score=clickbait_score,
        ai_score=ai_score,
        source_trust=source_trust,
        verification_results=verification_results,
        summary=summary,
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
