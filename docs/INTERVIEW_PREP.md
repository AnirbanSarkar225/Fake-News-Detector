# TruthShield — Technical Interview Preparation Guide

This guide is designed for technical interviewers, engineering hiring managers, and candidates to evaluate the engineering decisions, machine learning architecture, and system trade-offs behind **TruthShield**.

---

## 1. Project Overview & Architecture Summary

* **Primary Purpose:** Explainable credibility and misinformation analysis platform combining statistical NLP, ensemble machine learning, external fact-checking APIs, and explainable AI (XAI).
* **Corpus & Split:** 128,769 total news articles. Partitioned into an 80% training set (103,015 samples) and a strictly held-out 20% test set (25,754 samples).
* **Core Model:** Voting ensemble combining `LinearSVC`, `LogisticRegression`, `PassiveAggressiveClassifier`, and `ExtraTreesClassifier` over 75,000 TF-IDF features with n-gram range `(1, 3)`.
* **Empirical Test Results:** 89.92% Accuracy, 89.04% Precision, 93.36% Recall, 0.9115 F1-Score, 0.9652 ROC-AUC.
* **Serving Stack:** Streamlit for interactive human-in-the-loop inspection; FastAPI for programmatic headless JSON inference.

---

## 2. Key Engineering Decisions & Trade-Offs

| Decision | Alternative Considered | Engineering Rationale |
|---|---|---|
| **TF-IDF Ensemble over Pure LLM/BERT** | Zero-shot LLM inference (e.g. GPT-4 / Gemini for every query) | High token latency, high cost, and hallucination risks make pure LLMs impractical for real-time article-length vector processing. The classical TF-IDF ensemble processes a 2,000-word article in < 45ms with deterministic feature attributions. |
| **LinearSVC + PassiveAggressive in Ensemble** | Deep Neural Networks (LSTM/GRU) | Linear models are computationally efficient on ultra-sparse 75,000-dimensional TF-IDF vectors, converge rapidly, and allow direct weight inspection for explainability. |
| **Separation of NLP Classification from Fact Verification** | Treating NLP output as absolute factual truth | Stylometric models only detect linguistic patterns (sensationalism, hyperbolic vocabulary, syntactic anomalies). Authoritative factual debunking requires external corroboration (Google Fact Check API, Snopes), which TruthShield treats as a separate, parallel signal. |
| **Dual Interface (Streamlit + FastAPI)** | Monolithic Streamlit-only UI | While Streamlit offers rapid interactive prototyping with visual heatmaps, production microservices require headless JSON endpoints. Splitting logic into `utils/pipeline.py` allows both `app.py` and `api_server.py` to share identical core logic. |

---

## 3. 15 Technical Interview Questions & Grounded Answers

### Q1: Why did you choose an ensemble model instead of relying on a single top-performing classifier like LinearSVC?
**Answer:** While `LinearSVC` achieved the highest individual test accuracy (89.96%), it outputs uncalibrated decision-function distances rather than calibrated posterior probabilities. By ensembling `LinearSVC` with `LogisticRegression` (which outputs sigmoid-based probabilities), `PassiveAggressiveClassifier` (which handles noisy text boundaries well), and `ExtraTrees` (which captures non-linear feature interactions), the ensemble reduces individual model variance and delivers a significantly more robust probability distribution across edge-case articles.

### Q2: How did you select the TF-IDF parameters (75,000 max features, n-gram range 1–3)?
**Answer:** News articles rely heavily on phrasal collocations and contextual modifiers (e.g., *"breaking: claimed that"*, *"sources confirmed"*). Unigrams alone lose negation and modifier context, whereas bi-grams and tri-grams capture critical rhetorical phrasing. Limiting features to 75,000 prevents vocabulary explosion and out-of-memory errors during matrix factorization while maintaining top lexical discriminators identified via sublinear term-frequency scaling.

### Q3: How do you prevent data leakage between your training and testing partitions?
**Answer:** The `train_test_split` (80/20) is executed strictly before any feature extraction or vectorization. The `TfidfVectorizer` is fitted exclusively on `X_train` (`fit_transform`) and subsequently used only to `transform` `X_test`. This ensures that terms unique to the test partition cannot influence IDF weights or feature indices during training.

### Q4: What is the significance of the 93.36% Recall compared to the 89.04% Precision?
**Answer:** In misinformation filtering, the operational cost of false negatives (marking a fake or manipulative article as credible) is generally higher than false positives. A recall of 93.36% means that over 93 out of 100 credible news pieces are accurately identified, minimizing erroneous censorship, while the 89.04% precision ensures that articles flagged with high suspicion have an approximately 9 out of 10 likelihood of genuinely containing misleading patterns.

### Q5: How does TruthShield handle non-English or multilingual inputs?
**Answer:** The pipeline incorporates a language detection module (`langdetect`). If the input text is identified as non-English (such as Hindi or Bengali), it passes through a translation layer (`deep-translator`) into standardized English before tokenization and vectorization. For Indian vernacular contexts, experiments with MuRIL (Multilingual Representations for Indian Languages) were initiated in `scripts/train_muril.py`.

### Q6: Can a stylometric ML model detect factual falsehoods when an article is written in clean, journalistic prose?
**Answer:** No. Machine learning models operating over text vectors detect stylistic, lexical, and structural correlations—not objective ground truth. A well-written fake news article with standard journalistic syntax will evade TF-IDF classifiers. To counter this fundamental limitation, TruthShield decouples stylometry from factual verification by querying the Google Fact Check Tools API and cross-referencing domain reputation scores.

### Q7: How does sentence-level explainability work in the application?
**Answer:** Rather than presenting an opaque scalar score, the text is split into constituent sentences using NLTK sentence tokenization. Each sentence is independently evaluated through the vectorizer and classifier weights to compute a local suspicion index. Sentences exceeding the suspicion threshold are rendered in the Streamlit UI with highlighting and color gradients.

### Q8: What role does SHAP (SHapley Additive exPlanations) play in TruthShield?
**Answer:** SHAP computes the marginal contribution of individual vocabulary tokens to the final prediction log-odds based on cooperative game theory. It generates waterfall and summary plots indicating exactly which words pushed the score towards "Credible" versus "Suspicious", giving journalists transparent auditability into why a score was generated.

### Q9: How are news articles scraped from URLs, and how do you handle anti-scraping or paywalls?
**Answer:** URL extraction relies on `newspaper3k` supplemented by `BeautifulSoup4` for DOM traversal. User-Agent headers are randomized, and timeout thresholds are enforced. If an article is gated behind a hard paywall or dynamic client-side rendering (CAPTCHA/Cloudflare), the scraper gracefully catches the HTTP error and prompts the user to paste the raw text directly.

### Q10: How do you handle adversarial attempts to manipulate the classifier, such as "good word stuffing"?
**Answer:** In `testing/stress_test_hard.py`, adversarial test cases were introduced where fake stories were padded with reputable phrases (e.g., *"according to Reuters and WHO"*). While linear TF-IDF models can suffer from word-stuffing, TruthShield mitigates this by computing the ratio of headline sensationalism (`clickbait_detector.py`) and checking the domain origin (`source_engine.py`) rather than relying solely on body token frequencies.

### Q11: Why is ONNX Runtime export included in `scripts/export_onnx.py`?
**Answer:** For low-latency production deployment and cross-platform serving, ONNX (Open Neural Network Exchange) enables exporting scikit-learn models into an optimized runtime graph. Inference executed via ONNX Runtime reduces memory footprints and eliminates the overhead of full Python/scikit-learn runtime dependencies.

### Q12: How are external fact-checking claims matched to article contents?
**Answer:** In `utils/claim_verifier.py`, key named entities and subject-predicate phrases are extracted using SpaCy. These claims are queried against the Google Fact Check Tools API endpoint. If an exact or close semantic match is returned with an authoritative rating (e.g., "Pants on Fire", "False", "Verified"), the external verdict is prioritized over the probabilistic ML score.

### Q13: What is the memory footprint and cold-start latency of the model ensemble?
**Answer:** The serialized joblib model artifacts (`ensemble_model.joblib` and `tfidf_vectorizer.joblib`) occupy approximately 180MB on disk. During startup, they load into RAM within 1.2 to 2.5 seconds. Subsequent inference operations take between 25ms and 60ms per article on standard CPU hardware.

### Q14: How does the system prevent Server-Side Request Forgery (SSRF) when accepting arbitrary URLs?
**Answer:** The URL parsing utility validates protocol schemes (restricting to `http` and `https`), resolves DNS hostnames to verify they do not point to localhost (`127.0.0.1`), private RFC 1918 subnets (`10.0.0.0/8`, `192.168.0.0/16`), or internal cloud metadata IP addresses (`169.254.169.254`).

### Q15: What are the primary technical limitations of the current implementation?
**Answer:** 
1. Temporal staleness: The static TF-IDF vocabulary does not automatically adapt to newly emergent political entities or novel event jargon without retraining.
2. Contextual nuance: Sarcasm, irony, and parody require deeper contextual transformer representations (e.g. DeBERTa/RoBERTa) rather than n-gram bags of words.
3. Dependence on external API rate limits for the Google Fact Check endpoint.
