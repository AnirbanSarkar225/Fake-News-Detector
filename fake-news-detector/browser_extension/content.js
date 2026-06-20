// Content script that automatically scans articles and shows a credibility badge

(function() {
    'use strict';

    const API_BASE = 'http://localhost:8000';
    const MIN_TEXT_LENGTH = 200; // Don't analyze very short pages
    const BADGE_ID = 'truthshield-credibility-badge';
    const PANEL_ID = 'truthshield-analysis-panel';

    // Extract article text from the page
    function extractArticleText() {
        // Try semantic selectors first
        const selectors = [
            'article', '[role="article"]', '.article-body', '.article-content',
            '.post-content', '.entry-content', '.story-body', 'main',
            '.content-body', '#article-body', '.article__body'
        ];
        
        for (const sel of selectors) {
            const el = document.querySelector(sel);
            if (el && el.innerText.trim().length > MIN_TEXT_LENGTH) {
                return el.innerText.trim();
            }
        }

        // Fallback: get all paragraph text
        const paragraphs = document.querySelectorAll('p');
        const text = Array.from(paragraphs).map(p => p.innerText.trim()).filter(t => t.length > 20).join('\n');
        return text.length > MIN_TEXT_LENGTH ? text : null;
    }

    // Create floating credibility badge (bottom-right corner)
    function createBadge(result) {
        // Remove existing badge if any
        const existing = document.getElementById(BADGE_ID);
        if (existing) existing.remove();

        const badge = document.createElement('div');
        badge.id = BADGE_ID;
        
        const credibility = Math.round(result.credibility * 100);
        const prediction = result.prediction || 'UNKNOWN';
        const category = result.category || prediction;
        
        // Color based on credibility
        let bgColor, emoji, borderColor;
        if (credibility >= 70) {
            bgColor = '#065f46'; borderColor = '#10b981'; emoji = '✅';
        } else if (credibility >= 40) {
            bgColor = '#78350f'; borderColor = '#f59e0b'; emoji = '⚠️';
        } else {
            bgColor = '#7f1d1d'; borderColor = '#ef4444'; emoji = '❌';
        }

        badge.style.cssText = `
            position: fixed; bottom: 24px; right: 24px; z-index: 999999;
            background: ${bgColor}; border: 2px solid ${borderColor};
            border-radius: 16px; padding: 12px 20px; cursor: pointer;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: white; font-size: 14px; font-weight: 600;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4); backdrop-filter: blur(8px);
            display: flex; align-items: center; gap: 8px;
            transition: all 0.3s ease; user-select: none;
        `;
        badge.innerHTML = `
            <span style="font-size:20px">${emoji}</span>
            <span>TruthShield: ${credibility}% Credible</span>
            <span style="font-size:11px;opacity:0.7;margin-left:4px">${category}</span>
        `;

        badge.addEventListener('mouseenter', () => {
            badge.style.transform = 'scale(1.05)';
            badge.style.boxShadow = '0 12px 40px rgba(0,0,0,0.5)';
        });
        badge.addEventListener('mouseleave', () => {
            badge.style.transform = 'scale(1)';
            badge.style.boxShadow = '0 8px 32px rgba(0,0,0,0.4)';
        });

        // Click to expand panel
        badge.addEventListener('click', () => togglePanel(result));

        document.body.appendChild(badge);
    }

    // Create expandable analysis panel
    function togglePanel(result) {
        const existing = document.getElementById(PANEL_ID);
        if (existing) { existing.remove(); return; }

        const panel = document.createElement('div');
        panel.id = PANEL_ID;
        const cred = Math.round(result.credibility * 100);
        const conf = Math.round(result.confidence * 100);
        
        panel.style.cssText = `
            position: fixed; bottom: 80px; right: 24px; z-index: 999998;
            background: #111827; border: 1px solid #374151; border-radius: 16px;
            padding: 24px; width: 340px; max-height: 500px; overflow-y: auto;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            color: #e5e7eb; font-size: 13px; line-height: 1.6;
            box-shadow: 0 16px 48px rgba(0,0,0,0.5);
        `;

        let verificationHTML = '';
        if (result.verification_results && result.verification_results.length > 0) {
            verificationHTML = `
                <div style="margin-top:12px;padding-top:12px;border-top:1px solid #374151">
                    <div style="font-weight:600;margin-bottom:6px">🔍 Fact Verification</div>
                    ${result.verification_results.slice(0, 3).map(v => 
                        `<div style="font-size:12px;padding:4px 0">• ${v.claim_text || v.claim}: <span style="color:${String(v.rating).toLowerCase() === 'true' ? '#10b981' : String(v.rating).toLowerCase() === 'false' ? '#ef4444' : '#f59e0b'}">${v.rating || 'Unknown'}</span></div>`
                    ).join('')}
                </div>
            `;
        }

        panel.innerHTML = `
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px">
                <span style="font-weight:700;font-size:16px">🛡️ TruthShield Analysis</span>
                <span id="truthshield-close" style="cursor:pointer;font-size:18px;opacity:0.6">✕</span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px">
                <div style="background:#1f2937;border-radius:8px;padding:10px;text-align:center">
                    <div style="font-size:22px;font-weight:700;color:${cred >= 70 ? '#10b981' : cred >= 40 ? '#f59e0b' : '#ef4444'}">${cred}%</div>
                    <div style="font-size:11px;opacity:0.7">Credibility</div>
                </div>
                <div style="background:#1f2937;border-radius:8px;padding:10px;text-align:center">
                    <div style="font-size:22px;font-weight:700">${conf}%</div>
                    <div style="font-size:11px;opacity:0.7">Confidence</div>
                </div>
            </div>
            <div style="background:${result.prediction === 'REAL' ? '#065f46' : '#7f1d1d'};border-radius:8px;padding:8px 12px;text-align:center;font-weight:600;margin-bottom:12px">
                Verdict: ${result.category || result.prediction}
            </div>
            ${result.summary ? `<div style="font-size:12px;opacity:0.8;margin-bottom:8px">${result.summary}</div>` : ''}
            ${result.clickbait_score !== undefined ? `
                <div style="margin-top:8px;font-size:12px">
                    📰 Clickbait Score: ${Math.round(result.clickbait_score * 100)}%
                </div>
            ` : ''}
            ${result.ai_score !== undefined ? `
                <div style="font-size:12px">
                    🤖 AI-Generated Probability: ${Math.round(result.ai_score * 100)}%
                </div>
            ` : ''}
            ${result.source_trust !== undefined ? `
                <div style="font-size:12px">
                    🏛️ Source Trust: ${Math.round(result.source_trust)}%
                </div>
            ` : ''}
            ${verificationHTML}
            <div style="margin-top:12px;text-align:center">
                <a href="http://localhost:8501" target="_blank" style="color:#60a5fa;font-size:12px;text-decoration:none">
                    Open Full Analysis Dashboard →
                </a>
            </div>
        `;

        document.body.appendChild(panel);
        document.getElementById('truthshield-close').addEventListener('click', () => panel.remove());
    }

    // Show loading badge
    function showLoadingBadge() {
        const existing = document.getElementById(BADGE_ID);
        if (existing) existing.remove();

        const badge = document.createElement('div');
        badge.id = BADGE_ID;
        badge.style.cssText = `
            position: fixed; bottom: 24px; right: 24px; z-index: 999999;
            background: #1f2937; border: 2px solid #4b5563;
            border-radius: 16px; padding: 12px 20px;
            font-family: -apple-system, sans-serif;
            color: #9ca3af; font-size: 14px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            display: flex; align-items: center; gap: 8px;
        `;
        badge.innerHTML = '<span style="animation:spin 1s linear infinite;display:inline-block">⏳</span> Analyzing...';
        
        const style = document.createElement('style');
        style.textContent = '@keyframes spin { to { transform: rotate(360deg) } }';
        badge.appendChild(style);
        document.body.appendChild(badge);
    }

    // Main: auto-analyze on page load
    async function autoAnalyze() {
        const text = extractArticleText();
        if (!text) return; // Not an article page

        showLoadingBadge();

        try {
            const response = await fetch(`${API_BASE}/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text.substring(0, 10000), // Limit text size
                    url: window.location.href
                })
            });

            if (!response.ok) throw new Error(`API error: ${response.status}`);
            const result = await response.json();
            createBadge(result);
        } catch (err) {
            // Try simpler predict endpoint as fallback
            try {
                const response = await fetch(`${API_BASE}/predict-json`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: text.substring(0, 10000) })
                });
                if (response.ok) {
                    const result = await response.json();
                    createBadge(result);
                } else {
                    // Remove loading badge silently
                    const badge = document.getElementById(BADGE_ID);
                    if (badge) badge.remove();
                }
            } catch {
                const badge = document.getElementById(BADGE_ID);
                if (badge) badge.remove();
            }
        }
    }

    // Run after page load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => setTimeout(autoAnalyze, 1500));
    } else {
        setTimeout(autoAnalyze, 1500);
    }
})();
