// TruthShield Popup Script
// Tries /analyze endpoint first, falls back to /predict-json

(function () {
  'use strict';

  const API_BASE = 'http://localhost:8000';

  const newsTextEl = document.getElementById('newsText');
  const analyzeBtn = document.getElementById('analyzeBtn');
  const statusMsg = document.getElementById('statusMsg');
  const resultsArea = document.getElementById('resultsArea');

  // ── Initialization ──────────────────────────────────────────────

  // Check chrome.storage for text selected via context menu
  if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
    chrome.storage.local.get('selectedTextForVerify', (data) => {
      if (data.selectedTextForVerify) {
        newsTextEl.value = data.selectedTextForVerify;
        chrome.storage.local.remove('selectedTextForVerify');
        // Auto-analyze if text was injected from context menu
        runAnalysis();
      }
    });
  }

  analyzeBtn.addEventListener('click', runAnalysis);

  // Allow Ctrl+Enter to submit
  newsTextEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      runAnalysis();
    }
  });

  // ── Analysis Logic ──────────────────────────────────────────────

  async function runAnalysis() {
    const text = newsTextEl.value.trim();
    if (!text) {
      showStatus('Please paste article text or a URL to analyze.', true);
      return;
    }

    showStatus('Analyzing article...', false);
    analyzeBtn.disabled = true;
    resultsArea.classList.remove('visible');

    let result = null;

    // 1. Try the full /analyze endpoint
    try {
      const resp = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.substring(0, 10000), url: '' }),
      });
      if (resp.ok) {
        result = await resp.json();
      }
    } catch (_) {
      // Will fall through to fallback
    }

    // 2. Fallback to /predict-json
    if (!result) {
      try {
        const resp = await fetch(`${API_BASE}/predict-json`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: text.substring(0, 10000) }),
        });
        if (resp.ok) {
          result = await resp.json();
        }
      } catch (_) {
        // Will show error below
      }
    }

    analyzeBtn.disabled = false;

    if (!result) {
      showStatus('Could not reach TruthShield API. Make sure the server is running on localhost:8000.', true);
      return;
    }

    hideStatus();
    renderResults(result);
  }

  // ── Rendering ───────────────────────────────────────────────────

  function renderResults(r) {
    resultsArea.classList.add('visible');

    const prediction = r.prediction || 'UNKNOWN';
    const category = r.category || prediction;
    const credibility = r.credibility !== undefined ? Math.round(r.credibility * 100) : null;
    const confidence = r.confidence !== undefined ? Math.round(r.confidence * 100) : null;

    // Category badge
    const badgeEl = document.getElementById('categoryBadge');
    badgeEl.textContent = category;
    badgeEl.className = 'category-badge category-' + mapCategory(category);

    // Credibility score
    const credEl = document.getElementById('credibilityScore');
    if (credibility !== null) {
      credEl.textContent = credibility + '%';
      credEl.style.color = scoreColor(credibility);
    } else {
      credEl.textContent = '—';
    }

    // Confidence score card
    const confEl = document.getElementById('confidenceScore');
    confEl.textContent = confidence !== null ? confidence + '%' : '—';

    // Confidence bar
    if (confidence !== null) {
      document.getElementById('confidenceBarValue').textContent = confidence + '%';
      document.getElementById('confidenceBarFill').style.width = confidence + '%';
    }

    // Source trust
    const stIndicator = document.getElementById('sourceTrustIndicator');
    if (r.source_trust !== undefined && r.source_trust !== null) {
      const st = Math.round(r.source_trust);
      stIndicator.style.display = 'block';
      document.getElementById('sourceTrustValue').textContent = st + '%';
      document.getElementById('sourceTrustFill').style.width = st + '%';
    } else {
      stIndicator.style.display = 'none';
    }

    // Clickbait score
    const cbIndicator = document.getElementById('clickbaitIndicator');
    if (r.clickbait_score !== undefined && r.clickbait_score !== null) {
      const cb = Math.round(r.clickbait_score * 100);
      cbIndicator.style.display = 'block';
      document.getElementById('clickbaitValue').textContent = cb + '%';
      document.getElementById('clickbaitFill').style.width = cb + '%';
    } else {
      cbIndicator.style.display = 'none';
    }

    // AI-generated score
    const aiIndicator = document.getElementById('aiScoreIndicator');
    if (r.ai_score !== undefined && r.ai_score !== null) {
      const ai = Math.round(r.ai_score * 100);
      aiIndicator.style.display = 'block';
      document.getElementById('aiScoreValue').textContent = ai + '%';
      document.getElementById('aiScoreFill').style.width = ai + '%';
    } else {
      aiIndicator.style.display = 'none';
    }

    // Verification results
    const vSection = document.getElementById('verificationSection');
    const vList = document.getElementById('verificationList');
    if (r.verification_results && r.verification_results.length > 0) {
      vSection.style.display = 'block';
      vList.innerHTML = r.verification_results.slice(0, 5).map((v) => {
        const claimText = v.claim_text || v.claim || 'Claim';
        const rating = v.rating || 'Unknown';
        const ratingClass = getRatingClass(rating);
        return `<div class="verification-item">• ${escapeHtml(claimText)}: <span class="${ratingClass}">${escapeHtml(rating)}</span></div>`;
      }).join('');
    } else {
      vSection.style.display = 'none';
      vList.innerHTML = '';
    }
  }

  // ── Helpers ─────────────────────────────────────────────────────

  function mapCategory(cat) {
    const upper = (cat || '').toUpperCase();
    if (['REAL', 'TRUE', 'CREDIBLE', 'VERIFIED'].includes(upper)) return 'REAL';
    if (['FAKE', 'FALSE', 'FABRICATED'].includes(upper)) return 'FAKE';
    if (['SATIRE', 'PARODY'].includes(upper)) return 'SATIRE';
    if (['CLICKBAIT'].includes(upper)) return 'CLICKBAIT';
    if (['MISLEADING', 'BIASED', 'PARTIALLY FALSE'].includes(upper)) return 'MISLEADING';
    return 'UNKNOWN';
  }

  function scoreColor(val) {
    if (val >= 70) return '#10b981';
    if (val >= 40) return '#f59e0b';
    return '#ef4444';
  }

  function getRatingClass(rating) {
    const r = (rating || '').toLowerCase();
    if (r === 'true' || r === 'verified' || r === 'correct') return 'rating-true';
    if (r === 'false' || r === 'fabricated' || r === 'incorrect') return 'rating-false';
    return 'rating-mixed';
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function showStatus(msg, isError) {
    statusMsg.textContent = msg;
    statusMsg.className = 'status visible' + (isError ? ' error' : '');
  }

  function hideStatus() {
    statusMsg.className = 'status';
  }
})();
