"""
API Server for Fake News Detector Browser Extension
Provides REST endpoints for the browser extension to call for predictions
Runs on port 8000 by default
"""

import os
import sys
import joblib
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from utils.preprocess import TextPreprocessor

app = FastAPI(title="Fake News Detector API", version="1.0")

# Add CORS middleware to allow requests from browser extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (browser extension)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and preprocessor
model = None
preprocessor = None

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    credibility: float
    summary: str
    is_fake: bool

def load_model():
    """Load the trained model and preprocessor."""
    global model, preprocessor
    try:
        model_path = os.path.join(PROJECT_ROOT, "model", "fake_news_model.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            print("⚠️ Model file not found. Using fallback mode.")
            model = None
        
        preprocessor = TextPreprocessor()
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        preprocessor = TextPreprocessor()

def predict_article(text):
    """Run prediction on article text."""
    if not preprocessor or not model:
        # Fallback: return neutral response
        return {
            'prediction': 'UNCERTAIN',
            'confidence': 0.5,
            'credibility': 0.5,
            'summary': 'Model not available. Please use the web interface.',
            'is_fake': False
        }
    
    try:
        processed = preprocessor.preprocess_for_model(text)
        prediction = model.predict([processed])[0]
        
        try:
            decision_score = model.decision_function([processed])[0]
            confidence = min(abs(decision_score) / 3.0, 1.0)
        except Exception:
            confidence = 0.5
        
        if prediction == 'REAL':
            credibility = 0.5 + (confidence * 0.5)
            is_fake = False
        else:
            credibility = 0.5 - (confidence * 0.5)
            is_fake = True
        
        # Get top suspicious phrases for summary
        indicators = preprocessor.analyze_suspicious_indicators(text)
        summary = "No major red flags detected." if not is_fake else f"Detected {len(indicators.get('suspicious_indicators', []))} suspicious patterns."
        
        return {
            'prediction': prediction,
            'confidence': float(confidence),
            'credibility': float(credibility),
            'summary': summary,
            'is_fake': is_fake
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return {
            'prediction': 'ERROR',
            'confidence': 0.0,
            'credibility': 0.0,
            'summary': f'Prediction error: {str(e)}',
            'is_fake': False
        }

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Fake News Detector API is running"}

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict credibility of article text.
    
    Args:
        request: PredictRequest with 'text' field containing article text
    
    Returns:
        PredictResponse with prediction, confidence, and summary
    """
    if not request.text or len(request.text.strip()) < 20:
        raise HTTPException(status_code=400, detail="Text must be at least 20 characters long")
    
    result = predict_article(request.text)
    return PredictResponse(**result)

@app.post("/predict-json")
async def predict_json(request: PredictRequest):
    """Alternative endpoint returning plain JSON (for browser extension compatibility)."""
    if not request.text or len(request.text.strip()) < 20:
        return JSONResponse(
            status_code=400,
            content={"error": "Text must be at least 20 characters long"}
        )
    
    result = predict_article(request.text)
    return JSONResponse(content=result)

@app.get("/health")
async def health():
    """Health check endpoint with model status."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

if __name__ == "__main__":
    import uvicorn
    
    # Load model on startup
    print("🚀 Loading Fake News Detector model...")
    load_model()
    
    if model is None:
        print("⚠️ Warning: Model not loaded. API will use fallback mode.")
    else:
        print("✅ Model loaded successfully")
    
    print("🌐 Starting API server on http://localhost:8000")
    print("📍 API endpoint: POST /predict")
    print("📍 Documentation: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
