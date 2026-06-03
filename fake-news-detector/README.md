# 🛡️ Fake News & Misinformation Detector

An AI-powered tool that analyzes news articles and predicts their credibility using **Natural Language Processing (NLP)** and **Machine Learning**. Built with a Streamlit dashboard featuring sentence-level explainability.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- **📝 Paste Article** — Enter article text directly for instant analysis
- **🔗 URL Extraction** — Automatically scrape and analyze articles from any URL
- **🎯 Credibility Prediction** — ML-based fake/real classification with confidence scores
- **🔎 Explainability** — Sentence-level suspicious claim highlighting with color-coded severity
- **📊 Visual Dashboard** — Interactive gauge charts, metrics, and pattern breakdowns
- **⚡ Fast Inference** — Lightweight TF-IDF + PassiveAggressiveClassifier pipeline

---

## 📁 Project Structure

```
fake-news-detector/
├── app.py                  # Streamlit web application
├── train_model.py          # Model training script (optional)
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
├── model/
│   ├── fake_news_model.pkl # Pre-trained model (included)
│   └── preprocessor.pkl    # Saved preprocessor (included)
├── data/
│   └── (dataset files)     # Training dataset (only needed if retraining)
├── utils/
│   ├── __init__.py
│   ├── preprocess.py       # Text preprocessing, analysis & explainability
│   └── scraper.py          # URL article scraper using Newspaper3k
└── README.md
```

---

## 🚀 Quick Start

The trained model is **already included** — no dataset download or training required!

### Prerequisites

- **Python 3.9+** (tested with 3.11)
- **pip** or [**uv**](https://docs.astral.sh/uv/) (recommended for faster installs)

### Step 1: Get the Project

**Option A — Clone from GitHub:**

```bash
git clone https://github.com/AnirbanSarkar225/Project-News.git
cd Project-News/fake-news-detector
```

**Option B — Download from Google Drive:**

1. Download the zip from: **[Google Drive Link](https://drive.google.com/your-link-here)**
2. Extract the zip
3. Open a terminal in the extracted folder

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

> 💡 **Tip:** For faster installs, use [uv](https://docs.astral.sh/uv/):
> ```bash
> uv venv --python 3.11
> .venv\Scripts\activate       # Windows
> source .venv/bin/activate    # macOS/Linux
> uv pip install -r requirements.txt
> ```

### Step 3: Run the App

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501** 🚀

That's it! Paste any news article or enter a URL to check its credibility.

---

## 🔄 Retraining the Model (Optional)

If you want to retrain the model with the full dataset or your own data:

### 1. Download the Dataset

Download from Kaggle: **[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)**

Place the files in the `data/` directory:

| Format | Files Required |
|--------|---------------|
| **Kaggle original** | `data/Fake.csv` + `data/True.csv` |
| **Combined CSV** | `data/news.csv` (must have `text` and `label` columns) |

### 2. Train

```bash
python train_model.py
```

This will train the model and save it to `model/fake_news_model.pkl`, replacing the existing one.

---

## 🧠 How It Works

### ML Pipeline

```
Raw Text → Cleaning → TF-IDF Vectorization → PassiveAggressiveClassifier → Prediction
                                                                                ↓
                                                        Sentence Analysis → Explainability
```

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Text Preprocessing | NLTK | Cleaning, tokenization, stopword removal, lemmatization |
| Feature Extraction | TF-IDF | 50K features, unigrams + bigrams, sublinear TF |
| Classifier | PassiveAggressiveClassifier | Binary classification (FAKE / REAL) |
| URL Scraping | Newspaper3k | Extract article text and metadata from URLs |
| Explainability | Custom sentence scorer | Pattern detection + per-sentence model scoring |
| Frontend | Streamlit + Plotly | Interactive dashboard with gauge charts |

### Explainability Engine

The system provides transparency through three layers:

1. **Sentence-level ML scoring** — Each sentence is individually passed through the trained model to get a suspicion score
2. **Pattern detection** — Identifies sensationalist language (e.g., "BREAKING", "you won't believe"), excessive caps, vague sourcing
3. **Credibility indicators** — Flags positive signals like citations, institutional references, and data-backed claims

Sentences are color-coded in the dashboard:
- 🔴 **Red** — High suspicion (score > 50%)
- 🟡 **Yellow** — Medium suspicion (score 20–50%)
- 🟢 **Green** — Low suspicion (score < 20%)

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.9+ |
| ML Pipeline | scikit-learn (TF-IDF + PassiveAggressiveClassifier) |
| NLP | NLTK (tokenization, lemmatization, stopwords) |
| Article Extraction | Newspaper3k |
| Web Interface | Streamlit |
| Visualizations | Plotly |
| Data Processing | Pandas, NumPy |

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|---------|
| `ModuleNotFoundError` | Make sure dependencies are installed: `pip install -r requirements.txt` |
| `Model not found` warning in app | The model should be included. If missing, retrain with `python train_model.py` |
| NLTK data missing | The app auto-downloads required NLTK resources on first run |
| URL extraction fails | Some sites block scraping; try pasting the article text directly |
| Slow `pip install` | Use `uv pip install -r requirements.txt` instead — it's 10-100× faster |

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It should not be used as the sole source for determining the credibility of news articles. Always verify information with multiple trusted sources.

---

## 📜 License

MIT License — Feel free to use and modify for your projects.
