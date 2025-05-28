/// script.js
document.addEventListener('DOMContentLoaded', () => {
    const inputTextArea = document.getElementById('predict_info');
    const inputButton = document.getElementById('detect_button');
    const clearButton = document.getElementById('clear_button');
    const normalOrScam = document.getElementById('is_scam');
    const confidenceScoreSpan = document.getElementById('confidence_score');
    const suspiciousPhrasesDiv = document.getElementById('suspicious_phrases');
    const feedbackArea = document.getElementById('feedback_area');
    const feedbackCorrectBtn = document.getElementById('feedback_correct');
    const feedbackWrongBtn = document.getElementById('feedback_wrong');
    const feedbackStatus = document.getElementById('feedback_status');

    let lastPrediction = null;

    // 使用相對路徑，Vercel 會自動解析
    const API_URL = '/predict';
    const FEEDBACK_API = '/feedback';

    inputButton.addEventListener('click', async () => {
        const message = inputTextArea.value.trim();
        if (!message) {
            alert('請輸入您想檢測的訊息內容。');
            return;
        }

        normalOrScam.textContent = '檢測中...';
        normalOrScam.style.color = 'gray';
        confidenceScoreSpan.textContent = '計算中...';
        suspiciousPhrasesDiv.innerHTML = '<p>正在分析訊息，請稍候...</p>';
        feedbackArea.style.display = 'none';
        feedbackStatus.textContent = '';
        feedbackCorrectBtn.style.display = 'inline-block';
        feedbackWrongBtn.style.display = 'inline-block';

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: message }),
            });

            if (!response.ok) throw new Error(`伺服器錯誤: ${response.status} ${response.statusText}`);
            const data = await response.json();

            updateResults(data.status, data.confidence, data.suspicious_keywords);
            feedbackArea.style.display = 'block';
            lastPrediction = { text: message, model_status: data.status };
        } catch (error) {
            console.error('訊息檢測失敗:', error);
            alert(`訊息檢測失敗，請檢查後端服務。\n錯誤詳情: ${error.message}`);
            resetResults();
        }
    });

    clearButton.addEventListener('click', () => {
        inputTextArea.value = '';
        resetResults();
        feedbackArea.style.display = 'none';
        feedbackStatus.textContent = '';
    });

    feedbackCorrectBtn.addEventListener('click', () => submitFeedback('正確'));
    feedbackWrongBtn.addEventListener('click', () => submitFeedback('錯誤'));

    async function submitFeedback(user_feedback) {
        if (!lastPrediction) return;
        const payload = { ...lastPrediction, user_feedback };
        try {
            const res = await fetch(FEEDBACK_API, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const msg = await res.json();
            feedbackStatus.textContent = '✅ 感謝你的回饋！';
            feedbackCorrectBtn.style.display = 'none';
            feedbackWrongBtn.style.display = 'none';
        } catch (e) {
            feedbackStatus.textContent = '❌ 回饋提交失敗';
        }
    }

    function updateResults(isScam, confidence, suspiciousParts) {
        normalOrScam.textContent = isScam;
        confidenceScoreSpan.textContent = confidence;
        suspiciousPhrasesDiv.innerHTML = '';
        if (suspiciousParts && suspiciousParts.length > 0) {
            const ul = document.createElement('ul');
            suspiciousParts.forEach(phrase => {
                const li = document.createElement('li');
                li.textContent = phrase;
                ul.appendChild(li);
            });
            suspiciousPhrasesDiv.appendChild(ul);
        } else {
            suspiciousPhrasesDiv.innerHTML = '<p>沒有偵測到特別可疑的詞句。</p>';
        }
    }

    function resetResults() {
        normalOrScam.textContent = '待檢測';
        normalOrScam.style.color = 'inherit';
        confidenceScoreSpan.textContent = '待檢測';
        suspiciousPhrasesDiv.innerHTML = '<p>請輸入訊息並點擊「檢測！」按鈕。</p>';
    }
});