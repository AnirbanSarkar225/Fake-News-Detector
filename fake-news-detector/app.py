"""
Fake News & Misinformation Detector — Streamlit Application

A premium, interactive dashboard for detecting fake news articles.
Supports both direct text input and URL-based article extraction.
"""

import os
import sys
import joblib
import streamlit as st
import plotly.graph_objects as go
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.preprocess import TextPreprocessor
from utils.scraper import ArticleScraper

st.set_page_config(
    page_title="Fake News Detector — AI Credibility Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .hero-header {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b2d45 50%, #162a3e 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(78, 205, 196, 0.12);
        box-shadow: 0 20px 60px rgba(0,0,0,0.25);
    }
    .hero-header h1 {
        background: linear-gradient(135deg, #4ecdc4, #7c8cf0, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .hero-header p {
        color: #8ba3b8;
        font-size: 1.05rem;
        font-weight: 300;
    }

    .result-card {
        background: linear-gradient(145deg, #1b2d45, #213a54);
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1rem 0;
        border: 1px solid rgba(78, 205, 196, 0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.18);
    }
    .result-card h3 {
        color: #d0dce6;
        font-weight: 600;
        margin-bottom: 1rem;
        font-size: 1.15rem;
    }

    .verdict-real {
        background: linear-gradient(135deg, #3dbaa2, #4fd1a5);
        color: #0d1b2a;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
        box-shadow: 0 4px 20px rgba(61,186,162,0.3);
        letter-spacing: 1px;
    }
    .verdict-fake {
        background: linear-gradient(135deg, #e06060, #f07070);
        color: #fff;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
        box-shadow: 0 4px 20px rgba(224,96,96,0.3);
        letter-spacing: 1px;
    }
    .verdict-uncertain {
        background: linear-gradient(135deg, #e0a040, #f0b060);
        color: #0d1b2a;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.3rem;
        display: inline-block;
        box-shadow: 0 4px 20px rgba(224,160,64,0.3);
        letter-spacing: 1px;
    }

    .sentence-high {
        background: rgba(224,96,96,0.12);
        border-left: 3px solid #f07070;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 8px 8px 0;
        color: #d0dce6;
        font-size: 0.92rem;
    }
    .sentence-medium {
        background: rgba(240,176,96,0.10);
        border-left: 3px solid #f0b060;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 8px 8px 0;
        color: #d0dce6;
        font-size: 0.92rem;
    }
    .sentence-low {
        background: rgba(79,209,165,0.08);
        border-left: 3px solid #4fd1a5;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 8px 8px 0;
        color: #d0dce6;
        font-size: 0.92rem;
    }

    .metric-card {
        background: linear-gradient(145deg, #1b2d45, #243e58);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid rgba(78, 205, 196, 0.06);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4ecdc4, #7c8cf0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        color: #8ba3b8;
        font-size: 0.85rem;
        margin-top: 0.3rem;
        font-weight: 400;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b2d45 100%);
    }

    .stButton > button {
        background: linear-gradient(135deg, #3dbaa2, #5b8def) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2.5rem !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(61,186,162,0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(61,186,162,0.4) !important;
    }

    .info-box {
        background: rgba(78,205,196,0.08);
        border: 1px solid rgba(78,205,196,0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: #8ec8c0;
        font-size: 0.9rem;
    }

    .footer {
        text-align: center;
        color: #5e7a8f;
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid rgba(78, 205, 196, 0.08);
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model pipeline."""
    model_path = os.path.join("model", "fake_news_model.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


@st.cache_resource
def get_preprocessor():
    """Initialize text preprocessor."""
    return TextPreprocessor()


@st.cache_resource
def get_scraper():
    """Initialize article scraper."""
    return ArticleScraper()


def create_gauge_chart(score, title="Credibility Score"):
    """Create a beautiful gauge chart for the credibility score."""
    if score >= 0.7:
        bar_color = "#4fd1a5"
    elif score >= 0.4:
        bar_color = "#f0b060"
    else:
        bar_color = "#f07070"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 18, 'color': '#d0dce6', 'family': 'Inter'}},
        number={'suffix': '%', 'font': {'size': 42, 'color': '#d0dce6', 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#5e7a8f',
                     'tickfont': {'color': '#8ba3b8'}},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': '#1b2d45',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 35], 'color': 'rgba(224,96,96,0.12)'},
                {'range': [35, 65], 'color': 'rgba(240,176,96,0.10)'},
                {'range': [65, 100], 'color': 'rgba(79,209,165,0.08)'},
            ],
            'threshold': {
                'line': {'color': '#fff', 'width': 3},
                'thickness': 0.8,
                'value': score * 100
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=280,
        margin=dict(l=30, r=30, t=60, b=20),
        font={'family': 'Inter'}
    )
    return fig


def predict_article(text, model, preprocessor):
    """Run prediction and analysis on article text."""
    processed = preprocessor.preprocess_for_model(text)

    prediction = model.predict([processed])[0]

    try:
        decision_score = model.decision_function([processed])[0]
        confidence = min(abs(decision_score) / 3.0, 1.0)
    except Exception:
        confidence = 0.5

    if prediction == 'REAL':
        credibility = 0.5 + (confidence * 0.5)
    else:
        credibility = 0.5 - (confidence * 0.5)

    sentences = preprocessor.get_sentences(text)
    sentence_scores = []
    for sent in sentences[:20]:
        sent_processed = preprocessor.preprocess_for_model(sent)
        try:
            sent_decision = model.decision_function([sent_processed])[0]
            sent_suspicion = max(0, -sent_decision) / 3.0
            sent_suspicion = min(sent_suspicion, 1.0)
        except Exception:
            sent_suspicion = preprocessor.score_sentence_suspicion(sent)
        sentence_scores.append((sent, sent_suspicion))

    indicators = preprocessor.analyze_suspicious_indicators(text)

    return {
        'prediction': prediction,
        'confidence': confidence,
        'credibility': credibility,
        'sentence_analysis': sorted(sentence_scores, key=lambda x: x[1], reverse=True),
        'indicators': indicators,
    }


def main():
    st.markdown("""
    <div class="hero-header">
        <h1>🛡️ Fake News & Misinformation Detector</h1>
        <p>AI-powered credibility analysis using NLP & Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown("---")
        input_mode = st.radio(
            "Input Method",
            ["📝 Paste Article Text", "🔗 Enter URL"],
            index=0,
        )
        st.markdown("---")
        st.markdown("### 📊 About")
        st.markdown("""
        <div class="info-box">
            This tool uses <b>NLP</b> and <b>Machine Learning</b> to analyze news articles 
            and predict their credibility. It highlights suspicious claims and provides 
            an explainable breakdown of the analysis.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🔬 How It Works")
        st.markdown("""
        1. **Text Processing** — Clean & tokenize  
        2. **TF-IDF Vectorization** — Extract features  
        3. **ML Classification** — Predict credibility  
        4. **Explainability** — Highlight suspicious claims  
        """)
        st.markdown("---")
        st.markdown("""
        <div style="color:#5e7a8f;font-size:0.8rem;text-align:center;">
            Built with Streamlit, scikit-learn & Newspaper3k
        </div>
        """, unsafe_allow_html=True)

    model = load_model()
    preprocessor = get_preprocessor()
    scraper = get_scraper()

    if model is None:
        st.warning("⚠️ **Model not found!** Please train the model first.")
        st.markdown("""
        ### Quick Start:
        ```bash
        # 1. Download the Kaggle Fake/Real News dataset
        # 2. Place Fake.csv and True.csv in the data/ folder
        # 3. Train the model:
        python train_model.py
        # 4. Restart this app
        ```
        """)
        st.info("💡 The dataset is available at: [Kaggle — Fake and Real News Dataset]"
                "(https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)")
        return

    article_text = None

    if "📝" in input_mode:
        st.markdown("### 📝 Paste Your Article")
        article_text = st.text_area(
            "Enter the article text to analyze:",
            height=220,
            placeholder="Paste a news article here to check its credibility...",
            key="article_input"
        )
    else:
        st.markdown("### 🔗 Enter Article URL")
        url_input = st.text_input(
            "Article URL:",
            placeholder="https://www.example.com/news/article",
            key="url_input"
        )
        if url_input:
            with st.spinner("🔍 Extracting article from URL..."):
                result = scraper.extract_from_url(url_input)
                if result['success']:
                    article_text = result['text']
                    st.success(f"✅ Extracted: **{result['title']}**")
                    if result['authors']:
                        st.caption(f"Authors: {', '.join(result['authors'])}")
                    if result['publish_date']:
                        st.caption(f"Published: {result['publish_date']}")
                    with st.expander("📄 View Extracted Text", expanded=False):
                        st.text(article_text[:2000] + ("..." if len(article_text) > 2000 else ""))
                else:
                    st.error(f"❌ {result['error']}")

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        predict_clicked = st.button("🔍  Analyze Credibility", use_container_width=True, type="primary")

    if predict_clicked and article_text and len(article_text.strip()) > 50:
        with st.spinner("🧠 Analyzing article with AI..."):
            results = predict_article(article_text, model, preprocessor)

        st.markdown("---")

        st.markdown("## 📊 Analysis Results")

        col_verdict, col_gauge = st.columns([1, 1])

        with col_verdict:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("#### 🏷️ Verdict")

            pred = results['prediction']
            conf = results['confidence']

            if pred == 'REAL' and conf > 0.3:
                verdict_class = "verdict-real"
                verdict_text = "✅ LIKELY CREDIBLE"
                verdict_desc = "This article appears to be **credible** based on our analysis."
            elif pred == 'FAKE' and conf > 0.3:
                verdict_class = "verdict-fake"
                verdict_text = "🚨 LIKELY FAKE"
                verdict_desc = "This article shows signs of **misinformation** or fabrication."
            else:
                verdict_class = "verdict-uncertain"
                verdict_text = "⚠️ UNCERTAIN"
                verdict_desc = "The model is **uncertain** about this article. Verify with trusted sources."

            st.markdown(f'<div style="text-align:center;margin:1.5rem 0;">'
                        f'<span class="{verdict_class}">{verdict_text}</span></div>',
                        unsafe_allow_html=True)
            st.markdown(verdict_desc)
            st.markdown(f"**Model Confidence:** {conf*100:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_gauge:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            fig = create_gauge_chart(results['credibility'])
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        indicators = results['indicators']
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{indicators['sensationalism_score']*100:.0f}%</div>
                <div class="metric-label">Sensationalism</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{indicators['credibility_score']*100:.0f}%</div>
                <div class="metric-label">Credibility Signals</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{indicators['caps_ratio']*100:.0f}%</div>
                <div class="metric-label">Caps Ratio</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            word_count = len(article_text.split())
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{word_count:,}</div>
                <div class="metric-label">Word Count</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("#### 🔎 Sentence-Level Analysis")
        st.caption("Sentences are ranked by suspicion level — higher scores indicate more suspicious content.")

        for sent, score in results['sentence_analysis'][:10]:
            if score > 0.5:
                css_class = "sentence-high"
                icon = "🔴"
            elif score > 0.2:
                css_class = "sentence-medium"
                icon = "🟡"
            else:
                css_class = "sentence-low"
                icon = "🟢"

            display_sent = sent[:200] + "..." if len(sent) > 200 else sent
            st.markdown(
                f'<div class="{css_class}">{icon} <b>[{score*100:.0f}%]</b> {display_sent}</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

        if indicators['suspicious_patterns'] or indicators['credibility_indicators']:
            col_sus, col_cred = st.columns(2)

            with col_sus:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("#### ⚠️ Suspicious Patterns Found")
                if indicators['suspicious_patterns']:
                    for pat in set(indicators['suspicious_patterns'][:10]):
                        st.markdown(f"- 🚩 `{pat}`")
                else:
                    st.markdown("_No suspicious patterns detected._")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_cred:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("#### ✅ Credibility Indicators")
                if indicators['credibility_indicators']:
                    for pat in set(indicators['credibility_indicators'][:10]):
                        st.markdown(f"- ✅ `{pat}`")
                else:
                    st.markdown("_No credibility indicators found._")
                st.markdown('</div>', unsafe_allow_html=True)

    elif predict_clicked:
        st.warning("⚠️ Please enter at least 50 characters of article text to analyze.")

    st.markdown("""
    <div class="footer">
        <p>🛡️ Fake News Detector v1.0 — Built with Streamlit, scikit-learn & NLP</p>
        <p>⚠️ This tool is for educational purposes. Always verify information with trusted sources.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
