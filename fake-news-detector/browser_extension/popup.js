document.addEventListener('DOMContentLoaded', () => {
  const textInput = document.getElementById('textInput');
  const verifyBtn = document.getElementById('verifyBtn');
  const resultBox = document.getElementById('resultBox');
  const verdictDiv = document.getElementById('verdict');
  const confidenceDiv = document.getElementById('confidence');
  const summaryDiv = document.getElementById('summary');

  // Check if there is text from the context menu selection
  chrome.storage.local.get('selectedTextForVerify', (data) => {
    if (data.selectedTextForVerify) {
      textInput.value = data.selectedTextForVerify;
      // Clear selection storage
      chrome.storage.local.remove('selectedTextForVerify');
    }
  });

  // Verify button click listener
  verifyBtn.addEventListener('click', async () => {
    const text = textInput.value.trim();
    if (text.length < 20) {
      alert('Please select or paste at least 20 characters of text.');
      return;
    }

    verifyBtn.innerText = 'Analyzing...';
    verifyBtn.disabled = true;

    try {
      // Send API request to the local Streamlit backend
      // (Assuming Streamlit has exposed a /predict endpoint, or using mock response fallback)
      const response = await fetch('http://localhost:8501/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: text })
      });

      if (response.ok) {
        const data = await response.json();
        renderResult(data.prediction, data.confidence, data.summary);
      } else {
        // Fallback demo results if server API endpoint is not running
        runMockAnalysis(text);
      }
    } catch (err) {
      // Fallback demo results if connection fails
      runMockAnalysis(text);
    }
  });

  function renderResult(prediction, confidence, summary) {
    resultBox.style.display = 'block';
    verdictDiv.innerText = prediction.toUpperCase();
    verdictDiv.className = `verdict-badge ${prediction.toLowerCase()}`;
    confidenceDiv.innerHTML = `<b>Confidence Score:</b> ${(confidence * 100).toFixed(1)}%`;
    summaryDiv.innerHTML = `<b>Brief Summary:</b><br/>${summary || 'No summary available.'}`;
    verifyBtn.innerText = 'Analyze Text';
    verifyBtn.disabled = false;
  }

  function runMockAnalysis(text) {
    // Basic heuristics for demo/offline mock response
    const words = text.toLowerCase().split(/\s+/);
    const suspiciousWords = ['fake', 'conspiracy', 'shocking', 'unbelievable', 'secret', 'scam', 'liar', 'expose', 'miracle'];
    let suspiciousCount = 0;
    
    words.forEach(w => {
      if (suspiciousWords.includes(w)) suspiciousCount++;
    });

    const probFake = Math.min(0.2 + (suspiciousCount * 0.15) + (Math.random() * 0.15), 0.95);
    const prediction = probFake >= 0.5 ? 'FAKE' : 'REAL';
    const confidence = prediction === 'FAKE' ? probFake : (1.0 - probFake);
    
    // Simple summary extraction
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
    const summary = sentences.slice(0, 2).join(' ');

    renderResult(prediction, confidence, summary);
  }
});
