# 🛡️ Fake News Detector - Project Structure

## 📁 Organized Folder Structure

```
fake-news-detector/
├── 📂 src/                          # Source code
│   ├── app.py                       # Main Streamlit dashboard
│   └── api_server.py               # FastAPI server for browser extension
│
├── 📂 docs/                         # Documentation
│   ├── SETUP_GUIDE.md              # Detailed setup instructions
│   ├── QUICKSTART.md               # Quick 3-step startup guide
│   ├── IMPROVEMENTS.md             # Technical improvements details
│   ├── SYSTEM_SUMMARY.md           # Complete system overview
│   ├── CHANGES.md                  # Change index
│   └── README_FIXES.md             # Executive summary
│
├── 📂 scripts/                      # Utility scripts
│   ├── train_model.py              # Model training script
│   ├── download_and_combine.py     # Data download script
│   └── realtime_update.py          # Real-time fact-check updates
│
├── 📂 utils/                        # Core utilities
│   ├── preprocess.py               # Text preprocessing
│   ├── nlp_engine.py               # NLP analysis engine
│   ├── scraper.py                  # URL article scraper
│   ├── source_engine.py            # Source reputation analysis
│   └── pdf_generator.py            # PDF report generation
│
├── 📂 model/                        # ML models & metrics
│   ├── model_final.pkl             # Trained model
│   ├── preprocessor.pkl            # Text preprocessor
│   ├── evaluation_metrics.json      # Model performance metrics
│   └── README.md                   # Model documentation
│
├── 📂 data/                         # Dataset files
│   ├── Fake.csv                    # Fake news dataset
│   ├── True.csv                    # Real news dataset
│   ├── news.csv                    # Combined dataset
│   ├── truthshield.db              # User feedback database
│   └── README.md                   # Data documentation
│
├── 📂 browser_extension/            # Chrome/Firefox extension
│   ├── manifest.json               # Extension configuration
│   ├── popup.html                  # Extension popup UI
│   ├── popup.js                    # Extension script
│   └── background.js               # Background worker
│
├── 📂 assets/                       # Static assets
│   ├── logo.png                    # Application logo
│   └── smtp_config.json            # Email configuration
│
├── 📂 config/                       # Configuration files
│   ├── pyrightconfig.json          # Python type checking config
│   └── .streamlit/                 # Streamlit settings
│
├── requirements.txt                 # Python dependencies
├── README.md                        # Main project README
└── .gitignore                       # Git ignore rules

```

---

## 🚀 How to Run (Updated for New Structure)

### Step 1: Install Dependencies
```bash
cd d:\Project-News\fake-news-detector
pip install -r requirements.txt
```

### Step 2: Start API Server (Terminal 1)
```bash
cd d:\Project-News\fake-news-detector
python src/api_server.py
```

✅ Should see: `🌐 Starting API server on http://localhost:8000`

### Step 3: Start Streamlit App (Terminal 2)
```bash
cd d:\Project-News\fake-news-detector
streamlit run src/app.py
```

✅ Should see: `Local URL: http://localhost:8501`

---

## 📋 What's in Each Folder

| Folder | Purpose | Key Files |
|--------|---------|-----------|
| **src/** | Main application source code | `app.py`, `api_server.py` |
| **docs/** | Complete documentation | Setup guides, improvements, changes |
| **scripts/** | Utility & training scripts | `train_model.py`, `realtime_update.py` |
| **utils/** | Reusable utility modules | NLP, preprocessing, scraping |
| **model/** | ML models & metrics | `model_final.pkl`, evaluation data |
| **data/** | Datasets & user feedback | CSV files, SQLite database |
| **browser_extension/** | Chrome/Firefox extension | Manifest, popup UI, scripts |
| **assets/** | Images & configurations | Logo, email settings |
| **config/** | Configuration files | Streamlit, type checking |

---

## 📖 Documentation Location

All documentation has been moved to `docs/` folder:

- **Getting Started** → `docs/QUICKSTART.md`
- **Detailed Setup** → `docs/SETUP_GUIDE.md`
- **What Changed** → `docs/CHANGES.md` or `docs/IMPROVEMENTS.md`
- **System Overview** → `docs/SYSTEM_SUMMARY.md`
- **Executive Summary** → `docs/README_FIXES.md`

---

## 🔧 Database Location

User feedback database: `data/truthshield.db`

**Auto-created on first app run** - No manual setup needed

---

## 🎯 Import Updates

If you're importing from the utils, paths remain the same:
```python
from utils.preprocess import TextPreprocessor
from utils.nlp_engine import NLPEngine
from utils.scraper import ArticleScraper
from utils.source_engine import SourceEngine
```

---

## 💾 Quick Command Reference

```bash
# Run main app
python src/app.py

# Run API server
python src/api_server.py

# Train new model
python scripts/train_model.py

# Download/update data
python scripts/download_and_combine.py

# Get real-time fact checks
python scripts/realtime_update.py
```

---

## ✅ Benefits of New Structure

✅ **Better Organization** - Clear separation of concerns  
✅ **Easier Navigation** - Find what you need quickly  
✅ **Professional Layout** - Standard Python project structure  
✅ **Scalable** - Easy to add more modules  
✅ **Maintainable** - Clear documentation location  

---

**Project Status**: ✅ Reorganized & Ready!

Version: 2.0 (With Organized Folder Structure)
