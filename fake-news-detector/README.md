<p align="center">
  <img src="assets/logo.png" width="100" alt="TruthShield Logo" style="border-radius: 50%;">
</p>

<h1 align="center">🛡️ TruthShield — AI Credibility Intelligence Platform</h1>

<p align="center">
  <strong>Enterprise-grade credibility analysis powered by multi-model NLP, real-time fact-checking, and explainable AI.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/scikit--learn-1.5+-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Transformers-4.36+-FFD21E?logo=huggingface&logoColor=black" alt="HuggingFace">
  <img src="https://img.shields.io/badge/Model_Accuracy-89.6%25-42D392" alt="Accuracy">
  <img src="https://img.shields.io/badge/Corpus-128K+_articles-4EA8FF" alt="Corpus">
  <img src="https://img.shields.io/badge/License-MIT-E7C65A" alt="License">
</p>

---

## What Is TruthShield?

TruthShield is a full-stack AI platform that analyzes news articles and predicts their credibility using an ensemble of NLP models, source reputation databases, claim verification APIs, and linguistic heuristics. It provides **sentence-level explainability**, **SHAP feature attribution**, and **evidence-based verdicts** — not just binary labels.

Built for analysts, journalists, researchers, and anyone who wants to cut through misinformation.

### Key Capabilities

| Capability | Description |
|-----------|-------------|
| 🧠 **Multi-Model Ensemble** | TF-IDF + PassiveAggressive, DistilBERT, MuRIL (Indian languages), clickbait detector, AI-generated text detector, satire detector, stance analyzer |
| 🔍 **Real-Time Fact-Checking** | Google Fact Check API integration, Snopes/PolitiFact cross-referencing, claim extraction & verification |
| 🌐 **Source Intelligence** | Domain reputation scoring, political bias detection, MBFC cross-referencing, WHOIS age analysis |
| 📊 **Explainable AI** | SHAP waterfall charts, sentence-level suspicion scoring, pattern-based red flag detection |
| 🇮🇳 **Multilingual Support** | Hindi, Bengali, Tamil, Telugu, Gujarati, Marathi, Kannada, Malayalam, Punjabi via MuRIL + deep-translator |
| 📝 **PDF Intelligence Reports** | Downloadable credibility reports with evidence, scores, and analysis breakdown |
| 🔗 **Browser Extension** | Chrome extension for one-click credibility checks on any webpage |
| 📡 **Live News Feed** | Real-time Indian news headlines from NDTV, TOI, The Hindu via RSS |
| 🖼️ **Image Verification** | Reverse image search and OCR text extraction from news images |
| 📈 **Analytics Dashboard** | Category distribution, confidence trends, geographic analysis, model evaluation metrics |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TruthShield Intelligence Platform               │
├──────────────┬──────────────┬───────────────┬───────────────────────┤
│  Input Layer │  Processing  │  Verification │   Intelligence Output │
├──────────────┼──────────────┼───────────────┼───────────────────────┤
│ • Paste Text │ • TF-IDF     │ • Google      │ • Credibility Score   │
│ • URL Scrape │   Ensemble   │   Fact Check  │ • Risk Classification │
│ • Image OCR  │ • DistilBERT │ • Domain Rep  │ • SHAP Explanations   │
│ • News Feed  │ • Clickbait  │ • MBFC Bias   │ • Sentence Analysis   │
│ • Browser    │ • AI Detect  │ • Claim       │ • Evidence Grid       │
│   Extension  │ • Satire     │   Extraction  │ • Knowledge Graph     │
│              │ • Stance     │ • WHOIS       │ • PDF Report          │
│              │ • NLP Engine │               │ • Timeline            │
└──────────────┴──────────────┴───────────────┴───────────────────────┘
```

### ML Pipeline

```
Raw Article
   ↓
Language Detection → Translation (if non-English)
   ↓
Text Preprocessing (NLTK: tokenize, lemmatize, clean)
   ↓
┌─────────────────────────────────────────────────┐
│              Parallel Analysis                   │
│  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ TF-IDF +     │  │ Clickbait Detector       │ │
│  │ Classifier   │  │ AI-Generated Detector    │ │
│  │ Ensemble     │  │ Satire Detector          │ │
│  └──────────────┘  │ Stance Analyzer          │ │
│                     │ Source Intelligence       │ │
│                     │ Claim Verification (API)  │ │
│                     └──────────────────────────┘ │
└─────────────────────────────────────────────────┘
   ↓
Weighted Score Aggregation → Credibility Verdict
   ↓
SHAP Explainability + Sentence-Level Analysis
   ↓
Intelligence Report (Dashboard / PDF)
```

---

## Quick Start

The trained model is **already included** — no dataset download or training required.

### Prerequisites

- **Python 3.9+** (tested with 3.11, 3.14)
- **pip** or [**uv**](https://docs.astral.sh/uv/) (recommended)

### 1. Clone & Install

```bash
git clone https://github.com/AnirbanSarkar225/Fake-News-Detector.git
cd Fake-News-Detector/fake-news-detector
```

```bash
# Option A: pip
pip install -r requirements.txt

# Option B: uv (10-100× faster)
uv venv --python 3.11
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS/Linux
uv pip install -r requirements.txt
```

### 2. Configure Environment (Optional)

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

| Variable | Purpose | Required? |
|----------|---------|-----------|
| `GOOGLE_FACTCHECK_API_KEY` | Real-time claim verification via Google Fact Check Tools API | Optional (degrades gracefully) |
| `SMTP_USER` | Gmail address for OTP delivery | Optional (falls back to console OTP) |
| `SMTP_PASSWORD` | Gmail App Password for SMTP | Optional |
| `SMTP_SERVER` | SMTP server (default: `smtp.gmail.com`) | Optional |
| `SMTP_PORT` | SMTP port (default: `587`) | Optional |

> 💡 The app works without any API keys. Fact-checking and OTP email will simply fall back to offline/console modes.

### 3. Launch

```bash
# Full platform (API + Dashboard)
python start.py

# Dashboard only
streamlit run src/app.py

# API server only
python start.py --api
```

The app will open at **http://localhost:8501** 🚀

---

## Project Structure

```
fake-news-detector/
├── src/
│   └── app.py                      # Main Streamlit application (4,900+ lines)
├── utils/
│   ├── preprocess.py               # Text preprocessing & feature engineering
│   ├── nlp_engine.py               # NLP analysis: sentiment, entities, SHAP, summary
│   ├── scraper.py                  # URL article extraction (Newspaper3k)
│   ├── source_engine.py            # Domain reputation, MBFC bias, WHOIS analysis
│   ├── claim_verifier.py           # Google Fact Check API + claim extraction
│   ├── clickbait_detector.py       # Clickbait pattern classification
│   ├── ai_detector.py              # AI-generated text detection
│   ├── satire_detector.py          # Satire/parody content detection
│   ├── stance_detector.py          # Article stance analysis (supports/refutes)
│   ├── evidence_engine.py          # Evidence aggregation from multiple sources
│   ├── multilingual.py             # Language detection & translation (9 Indian langs)
│   ├── transformer_predictor.py    # DistilBERT / MuRIL transformer inference
│   ├── bert_predictor.py           # BERT-based prediction utilities
│   ├── pipeline.py                 # Unified prediction pipeline orchestrator
│   ├── india_news_feed.py          # Live RSS feed parser (NDTV, TOI, The Hindu)
│   ├── image_verifier.py           # Reverse image search + OCR
│   └── pdf_generator.py           # PDF intelligence report generator
├── scripts/
│   ├── train_model.py              # TF-IDF ensemble training (128K+ articles)
│   ├── train_bert.py               # DistilBERT fine-tuning script
│   ├── train_muril.py              # MuRIL fine-tuning for Indian languages
│   ├── export_onnx.py              # ONNX Runtime export for fast inference
│   ├── download_and_combine.py     # Dataset download & merge utility
│   ├── augment_training_data.py    # Data augmentation pipeline
│   ├── retrain_from_feedback.py    # Retrain model using user feedback
│   ├── realtime_update.py          # Background daemon for live data streams
│   └── seed_fake_domains.py        # Seed domain reputation database
├── model/
│   ├── fake_news_model.pkl         # Pre-trained TF-IDF ensemble (~15 MB)
│   ├── preprocessor.pkl            # Saved text preprocessor
│   └── evaluation_metrics.json     # Model performance metrics
├── data/
│   ├── news.csv                    # Combined training corpus (128K+ articles)
│   └── truthshield.db              # SQLite database (history, feedback, domains)
├── browser_extension/
│   ├── manifest.json               # Chrome extension manifest (MV3)
│   ├── popup.html                  # Extension popup UI
│   ├── popup.js                    # Extension popup logic
│   ├── content.js                  # Page content extraction
│   └── background.js               # Service worker
├── assets/
│   └── logo.png                    # TruthShield logo
├── config/                          # Runtime configuration files
├── .streamlit/
│   └── config.toml                 # Streamlit theme configuration
├── .env.example                     # Environment variable template
├── requirements.txt                 # Python dependencies (34 packages)
├── start.py                        # One-click launcher (API + Dashboard)
└── README.md
```

---

## Training & Datasets

### Pre-Trained Model (Included)

The included model was trained on **128,369 articles** across 5 datasets:

| Dataset | Articles | Source |
|---------|----------|--------|
| **ISOT Fake News** | ~45,000 | Reuters real articles + unreliable sources |
| **LIAR** | ~10,000 | PolitiFact statements with 6-way labels |
| **COVID-19 Misinformation** | ~8,500 | Pandemic claims & fact-checks |
| **McIntire** | ~6,300 | Benchmark fake/real news articles |
| **IFND (Indian)** | ~5,000 | Indian fact-checked news articles |
| **Fact Check API Streams** | ~53,500 | Daily automated collections |

**Performance Metrics:**

| Metric | Score |
|--------|-------|
| Accuracy | **89.62%** |
| Precision | 88.86% |
| Recall | 93.04% |
| F1 Score | 90.90% |

### Retraining (Optional)

```bash
# Retrain the TF-IDF ensemble
python scripts/train_model.py

# Fine-tune DistilBERT
python scripts/train_bert.py

# Fine-tune MuRIL (Indian languages)
python scripts/train_muril.py

# Export to ONNX for fast CPU inference
python scripts/export_onnx.py --model distilbert --quantize

# Retrain from user feedback
python scripts/retrain_from_feedback.py
```

---

## Browser Extension

The Chrome extension allows one-click credibility analysis on any webpage.

### Installation

1. Open `chrome://extensions/`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select the `browser_extension/` folder
5. Click the TruthShield icon on any news article

The extension connects to the local API server (`python start.py --api`) for analysis.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit, Plotly, Custom CSS (Premium dark theme) |
| **ML Core** | scikit-learn (TF-IDF + PassiveAggressiveClassifier ensemble) |
| **Deep Learning** | HuggingFace Transformers (DistilBERT, MuRIL), PyTorch, ONNX Runtime |
| **NLP** | NLTK, spaCy, textstat, SHAP, LIME |
| **Translation** | deep-translator, langdetect (9 Indian languages) |
| **Fact-Checking** | Google Fact Check Tools API, Wikipedia API |
| **Source Intelligence** | Custom domain reputation DB, MBFC cross-referencing, WHOIS |
| **Image Analysis** | EasyOCR, Pillow |
| **API Server** | FastAPI + Uvicorn |
| **Database** | SQLite (analysis history, user feedback, domain reputation) |
| **Reports** | ReportLab (PDF generation) |
| **Extension** | Chrome Manifest V3 |
| **Data** | Pandas, NumPy, BeautifulSoup, Newspaper3k, feedparser |

---

## Workspace Tabs

| Tab | Contents |
|-----|----------|
| 🔍 **Credibility Analyzer** | Article input, URL scraping, credibility verdict, evidence grid, knowledge graph, sentence analysis, SHAP explanations |
| 📖 **Media Literacy Hub** | SIFT method tutorial, bias checklist, fake news quiz, educational resources |
| 📊 **Analytics & Insights** | Category distribution pie chart, confidence trends, geographic analysis, domain statistics |
| 🔬 **Model Evaluation** | Confusion matrix, ROC curve, precision-recall, per-class metrics, retraining controls |
| 📋 **Analysis History** | Full history of past analyses with reload capability |

---

## Troubleshooting

| Issue | Solution |
|-------|---------|
| `ModuleNotFoundError` | Install dependencies: `pip install -r requirements.txt` |
| Model not found warning | The model is included. If missing, retrain: `python scripts/train_model.py` |
| NLTK data missing | Auto-downloaded on first run. Manual: `python -c "import nltk; nltk.download('punkt')"` |
| URL extraction fails | Some sites block scraping. Try pasting article text directly |
| spaCy model missing | Run: `python -m spacy download en_core_web_sm` |
| Slow `pip install` | Use `uv pip install -r requirements.txt` (10-100× faster) |
| SMTP/OTP not working | Set credentials in `.env` file, or check the console for the OTP code |
| Fact Check API errors | Get a free API key from [Google Cloud Console](https://console.cloud.google.com/) and add to `.env` |
| GPU not detected | CPU inference works fine. For GPU: install `torch` with CUDA support |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'Add your feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## Disclaimer

This tool is for **educational and research purposes only**. It should not be used as the sole source for determining the credibility of news articles. Always verify information with multiple trusted sources. The AI models may produce incorrect assessments, especially for topics outside the training data distribution.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://github.com/AnirbanSarkar225">Anirban Sarkar</a></sub>
</p>
