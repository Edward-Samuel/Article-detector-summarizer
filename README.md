# Article Detector & Summarizer Chrome Extension

A Chrome extension that uses advanced ML models to detect articles on web pages, extract their main content, and provide concise multilingual summaries.

## Features

- 🔍 **Intelligent Article Detection**: Uses multilingual BERT to accurately identify article content across languages
- 🌐 **Multilingual Support**: Handles 25+ languages with native language understanding
- 📝 **Smart Summarization**: Generates concise, bullet-pointed summaries using state-of-the-art mBART model
- 🎯 **Language-Specific Processing**: Custom formatting and handling for CJK, Arabic, Thai, and other languages
- 🎨 **Clean UI**: Modern, user-friendly interface
- 📋 **Copy Support**: Easy copying of both extracted content and summaries

## Technical Stack

### ML Models
- **Article Detection**: `bert-base-multilingual-cased`
  - Multilingual BERT model for content classification
  - Hybrid approach combining ML and structural analysis
  - Supports multiple languages and content types
  
- **Summarization**: `facebook/mbart-large-cc25`
  - Multilingual BART model trained on 25 languages
  - Language-aware summarization
  - Optimized for concise, meaningful summaries

### Backend
- Python 3.7+
- FastAPI for API server
- BeautifulSoup4 for HTML parsing
- PyTorch for ML operations
- Transformers library for BERT and mBART models

### Frontend
- Chrome Extension (Manifest V3)
- Vanilla JavaScript
- Modern CSS with Flexbox

## Supported Languages

The extension supports summarization in multiple languages including:
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Dutch (nl)
- Russian (ru)
- Japanese (ja)
- Korean (ko)
- Chinese (zh)
- Arabic (ar)
- Hindi (hi)
- Vietnamese (vi)
- Thai (th)
- Turkish (tr)
- Polish (pl)

## Installation

### Backend Setup

1. Clone the repository:
```bash
git clone https://github.com/[your-username]/article-detector-summarizer.git
cd article-detector-summarizer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the FastAPI server:
```bash
python api.py
```

The server will run on `http://localhost:8080`.

### Chrome Extension Setup

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" in the top right
3. Click "Load unpacked" and select the `extension` folder from this project

## Usage

1. Navigate to any webpage you want to analyze
2. Click the extension icon in your Chrome toolbar
3. The extension will:
   - Detect if the current page contains an article
   - Extract the main content (if it's an article)
   - Generate a multilingual summary with bullet points
4. Use the copy buttons to copy either the extracted content or the summary

## Project Structure

```
├── api.py                 # FastAPI server
├── text_summarizer.py     # ML models and text processing
├── requirements.txt       # Python dependencies
├── extension/            # Chrome extension files
│   ├── manifest.json     # Extension configuration
│   ├── popup.html       # Extension popup UI
│   ├── popup.js        # Extension logic
│   └── content.js      # Content script
└── README.md           # This file
```

## Model Details

### Article Detection Model (Multilingual BERT)
- Base model: bert-base-multilingual-cased
- Task: Binary classification (article/non-article)
- Features:
  - Multilingual support
  - HTML structure analysis
  - Content-based classification
  - Confidence threshold: 0.6

### Summarization Model (mBART)
- Base model: facebook/mbart-large-cc25
- Task: Multilingual text summarization
- Features:
  - Native support for 25 languages
  - Language-specific processing
  - Bullet point formatting
  - Custom prompt handling per language

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HuggingFace Transformers library
- BERT and mBART model creators
- FastAPI framework
- Chrome Extensions documentation 