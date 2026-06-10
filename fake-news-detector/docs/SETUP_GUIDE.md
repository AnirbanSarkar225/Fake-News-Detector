# 🛡️ Fake News Detector - Setup & Usage Guide

## Recent Improvements

### ✨ New Features

1. **Proper Feedback System** - User feedback is now saved to database and tracked
2. **API Server** - Separate FastAPI server for browser extension integration  
3. **Feedback Analytics** - Dashboard showing model misclassifications and user corrections
4. **Auto-correction Insights** - System identifies patterns in model errors for improvement
5. **Better Feedback Responses** - Different messages based on user agreement/rating levels

---

## 🚀 How to Run

### Prerequisites
- Python 3.8+
- All dependencies installed from `requirements.txt`

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Start the API Server (For Browser Extension)

Open a new terminal/PowerShell and run:

```bash
python api_server.py
```

Expected output:
```
🚀 Loading Fake News Detector model...
✅ Model loaded successfully
🌐 Starting API server on http://localhost:8000
📍 API endpoint: POST /predict
📍 Documentation: http://localhost:8000/docs
```

### Step 3: Start the Streamlit App

In another terminal/PowerShell, run:

```bash
streamlit run app.py
```

Expected output:
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

### Step 4: Access the Application

- **Web Dashboard**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Endpoint**: http://localhost:8000/predict

---

## 📊 Using the Feedback System

### In the Web Dashboard

1. **Analyze an Article** - Click "Analyze Credibility" button
2. **Review Results** - Check the model's prediction and confidence score
3. **Provide Feedback**:
   - Select: "Agree with AI", "Disagree with AI", or "Neutral"
   - Rate accuracy: 1-5 scale
   - Add optional notes explaining what the model got wrong
   - Click "Submit Feedback"

### Feedback Impact

- **Disagreements are flagged** - Shows warning and requests more analysis
- **Low ratings trigger alerts** - Model recalibration notice
- **Analytics Dashboard** - View all feedback and model corrections in the Analytics tab
- **Misclassification tracking** - See which articles the model got wrong

---

## 🔧 API Endpoints

### POST /predict
Predict credibility of article text.

**Request:**
```json
{
  "text": "Article text here..."
}
```

**Response:**
```json
{
  "prediction": "REAL",
  "confidence": 0.85,
  "credibility": 0.92,
  "summary": "No major red flags detected.",
  "is_fake": false
}
```

### POST /predict-json
Same as `/predict` but returns plain JSON (used by browser extension).

### GET /health
Check API and model status.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "preprocessor_loaded": true
}
```

### GET /
Health check endpoint.

---

## 🐛 Troubleshooting

### Issue: "API is not running"
**Solution**: Make sure you've started the API server in step 2:
```bash
python api_server.py
```

### Issue: "Model is classifying everything as fake"
**Solutions**:
1. Check that the model file exists at `model/model_final.pkl`
2. Provide feedback on misclassifications to help the system learn
3. Review the model training data in `data/` folder
4. Check feedback analytics to see patterns in errors

### Issue: "Feedback not being saved"
**Solution**: Check that the database is accessible:
```bash
# Database file should be at: data/truthshield.db
# Make sure data/ folder has write permissions
```

### Issue: Browser extension not working
**Solutions**:
1. Verify API server is running on port 8000
2. Check browser console for errors (F12 → Console tab)
3. Ensure no firewall is blocking localhost:8000
4. Try the fallback mock analysis if API is unavailable

---

## 📈 Improving Model Accuracy

The system learns from your feedback:

1. **Regular Feedback** - Provide feedback on every analysis
2. **Detailed Notes** - Explain why the model was wrong
3. **Rating Honestly** - Be accurate with 1-5 ratings
4. **Review Analytics** - Check feedback analytics to see patterns

Based on feedback data, the system identifies:
- Common false positives (real articles marked as fake)
- Common false negatives (fake articles marked as real)
- Specific topics where model struggles
- Patterns in misclassifications

---

## 📁 Project Structure

```
fake-news-detector/
├── app.py                  # Main Streamlit application
├── api_server.py          # FastAPI server for browser extension
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── utils/
│   ├── nlp_engine.py     # NLP analysis
│   ├── preprocess.py     # Text preprocessing
│   ├── scraper.py        # URL extraction
│   └── source_engine.py  # Source reputation analysis
├── model/
│   ├── model_final.pkl   # Trained model (must be present)
│   └── evaluation_metrics.json
├── data/
│   ├── truthshield.db    # User feedback database (auto-created)
│   ├── Fake.csv
│   ├── True.csv
│   └── news.csv
└── browser_extension/
    ├── manifest.json
    ├── popup.html
    └── popup.js
```

---

## 🎯 Common Workflows

### Workflow 1: Quick Web Analysis
1. Open http://localhost:8501
2. Sign in with Gmail (OTP via email or dev mode)
3. Paste article or enter URL
4. Click "Analyze Credibility"
5. Review results and provide feedback

### Workflow 2: Browser Extension Analysis
1. Install extension in Chrome/Firefox
2. Select text on any website
3. Click extension icon
4. View instant credibility prediction
5. (API server must be running)

### Workflow 3: Monitor Model Performance
1. Open Analytics tab in dashboard
2. View feedback analytics
3. Check misclassified articles
4. Note patterns in errors
5. Use this data to improve future model training

---

## 🔐 Security Notes

- **Model Loading**: Uses local model file, no external API calls for prediction
- **Database**: SQLite database stored locally in `data/` folder
- **SMTP**: Email credentials stored in `assets/smtp_config.json` (keep secure!)
- **API**: FastAPI server runs locally (http://localhost:8000)

---

## 📝 Notes for Developers

- The feedback system is designed to track model errors systematically
- `get_feedback_stats()` aggregates feedback by prediction type and user verdict
- `get_misclassified_articles()` returns articles where user disagreed with model
- Feedback data can be exported for fine-tuning or retraining the model
- API server uses CORS to allow requests from browser extension

---

**Last Updated**: 2026-06-10
**Version**: 1.1 (With Feedback System & API)
