# Article Detector & Summarizer Chrome Extension

A Chrome extension that uses machine learning to detect articles on web pages, extract their main content while removing ads and boilerplate, and provide concise summaries.

## Features

- 🔍 **Smart Article Detection**: Uses BERT model to identify article content
- 📝 **Content Extraction**: Removes ads, navigation, footers, and other non-article content
- 📚 **Text Summarization**: Generates concise summaries using T5 transformer model
- 🎨 **Clean UI**: Modern, user-friendly interface
- 📋 **Copy Support**: Easy copying of both extracted content and summaries

## Technical Stack

### Backend
- Python 3.7+
- FastAPI for the API server
- Transformers (BERT & T5) for ML tasks
- BeautifulSoup4 for HTML parsing
- PyTorch for ML operations

### Frontend
- Chrome Extension (Manifest V3)
- Vanilla JavaScript
- Modern CSS with Flexbox

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
   - Detect if the current page is an article
   - Extract the main content (if it's an article)
   - Generate a summary (if it's an article)
4. Use the copy buttons to copy either the extracted content or the summary

## Project Structure

```
├── api.py                 # FastAPI server
├── article_classifier.py  # Article detection & content extraction
├── text_summarizer.py    # Text summarization model
├── requirements.txt      # Python dependencies
├── extension/           # Chrome extension files
│   ├── manifest.json    # Extension configuration
│   ├── popup.html      # Extension popup UI
│   ├── popup.js        # Extension logic
│   └── content.js      # Content script
└── README.md           # This file
```

## Model Details

- **Article Detection**: Fine-tuned BERT model for binary classification
- **Content Extraction**: Rule-based + ML hybrid approach using BeautifulSoup4
- **Summarization**: T5 transformer model optimized for summarization

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
- BERT and T5 model creators
- FastAPI framework
- Chrome Extensions documentation 