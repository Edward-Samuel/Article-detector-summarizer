document.addEventListener('DOMContentLoaded', function() {
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    const status = document.getElementById('status');
    const content = document.getElementById('content');
    const summary = document.getElementById('summary');
    const copyContent = document.getElementById('copy-content');
    const copySummary = document.getElementById('copy-summary');
    const copySuccess = document.getElementById('copy-success');
    const contentSection = document.getElementById('content-section');
    const summarySection = document.getElementById('summary-section');
    const contentLang = document.getElementById('content-lang');
    const summaryLang = document.getElementById('summary-lang');

    // Language name mapping
    const languageNames = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'nl': 'Dutch',
        'ru': 'Russian',
        'ja': 'Japanese',
        'ko': 'Korean',
        'zh': 'Chinese',
        'ar': 'Arabic',
        'hi': 'Hindi'
    };

    function getLanguageName(code) {
        return languageNames[code] || code;
    }

    // Copy functionality
    function showCopySuccess() {
        copySuccess.style.opacity = '1';
        setTimeout(() => {
            copySuccess.style.opacity = '0';
        }, 2000);
    }

    async function copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            showCopySuccess();
        } catch (err) {
            console.error('Failed to copy text: ', err);
        }
    }

    copyContent.addEventListener('click', () => {
        copyToClipboard(content.textContent);
    });

    copySummary.addEventListener('click', () => {
        copyToClipboard(summary.textContent);
    });

    // Show loading state
    loading.style.display = 'block';
    result.style.display = 'none';
    contentSection.style.display = 'none';
    summarySection.style.display = 'none';

    // Get the current tab's HTML content
    chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        chrome.scripting.executeScript({
            target: {tabId: tabs[0].id},
            function: () => document.documentElement.outerHTML
        }, function(results) {
            if (chrome.runtime.lastError) {
                showError('Cannot access page content');
                return;
            }

            const htmlContent = results[0].result;
            
            // Send to backend for processing
            fetch('http://localhost:8080/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    html_content: htmlContent
                })
            })
            .then(response => response.json())
            .then(data => {
                loading.style.display = 'none';
                result.style.display = 'block';

                if (data.is_article) {
                    status.textContent = 'This page is an article';
                    status.className = 'status article';
                    
                    content.textContent = data.content;
                    contentSection.style.display = 'block';
                    
                    if (data.language) {
                        const langName = getLanguageName(data.language);
                        contentLang.textContent = langName;
                        summaryLang.textContent = langName;
                    }
                    
                    if (data.summary) {
                        summary.textContent = data.summary;
                        summarySection.style.display = 'block';
                    } else {
                        summarySection.style.display = 'none';
                    }
                } else {
                    status.textContent = 'This page is not an article';
                    status.className = 'status not-article';
                    contentSection.style.display = 'none';
                    summarySection.style.display = 'none';
                }
            })
            .catch(error => {
                showError('Error processing content: ' + error.message);
            });
        });
    });

    function showError(message) {
        loading.style.display = 'none';
        result.style.display = 'block';
        status.textContent = message;
        status.className = 'status not-article';
        contentSection.style.display = 'none';
        summarySection.style.display = 'none';
    }
}); 