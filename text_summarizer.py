from transformers import MBartForConditionalGeneration, MBartTokenizer, BertForSequenceClassification, BertTokenizer
import torch
from langdetect import detect
import re
from bs4 import BeautifulSoup

class ArticleDetector:
    def __init__(self):
        # Initialize multilingual BERT for article detection
        self.model_name = "bert-base-multilingual-cased"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def preprocess_text(self, html_content):
        # Parse HTML and extract text
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
            
        # Get text
        text = soup.get_text()
        
        # Clean text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text

    @torch.no_grad()
    def is_article(self, html_content):
        # Extract and preprocess text
        text = self.preprocess_text(html_content)
        
        # Check basic indicators
        if len(text.split()) < 100:  # Too short to be an article
            return False, text
            
        # Check structural indicators
        soup = BeautifulSoup(html_content, 'html.parser')
        article_indicators = {
            'has_article_tag': bool(soup.find('article')),
            'has_paragraphs': len(soup.find_all('p')) > 3,
            'has_headings': bool(soup.find(['h1', 'h2', 'h3'])),
            'has_article_class': bool(soup.find(class_=lambda x: x and 'article' in str(x).lower())),
        }
        
        # If strong structural indicators, return True
        if sum(article_indicators.values()) >= 3:
            return True, text
        
        # Use BERT for classification
        inputs = self.tokenizer(text[:512], 
                              return_tensors="pt",
                              truncation=True,
                              padding=True).to(self.device)
        
        outputs = self.model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        is_article = probabilities[0][1].item() > 0.6
        
        return is_article, text

class TextSummarizer:
    def __init__(self):
        # Initialize mBART model for multilingual summarization
        self.model_name = "facebook/mbart-large-cc25"
        self.tokenizer = MBartTokenizer.from_pretrained(self.model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode for faster inference
        
        # Initialize article detector
        self.article_detector = ArticleDetector()
        
        # Language code mappings for mBART
        self.language_codes = {
            'en': 'en_XX',
            'es': 'es_XX',
            'fr': 'fr_XX',
            'de': 'de_DE',
            'it': 'it_IT',
            'pt': 'pt_XX',
            'nl': 'nl_XX',
            'ru': 'ru_RU',
            'ja': 'ja_XX',
            'ko': 'ko_KR',
            'zh': 'zh_CN',
            'ar': 'ar_AR',
            'hi': 'hi_IN',
            'vi': 'vi_VN',
            'th': 'th_TH',
            'tr': 'tr_TR',
            'pl': 'pl_PL'
        }
        
        # Language-specific prompts
        self.language_prompts = {
            'en': 'Summarize the following text:',
            'es': 'Resume el siguiente texto:',
            'fr': 'Résume le texte suivant:',
            'de': 'Fasse den folgenden Text zusammen:',
            'it': 'Riassumi il seguente testo:',
            'pt': 'Resuma o seguinte texto:',
            'nl': 'Vat de volgende tekst samen:',
            'ru': 'Обобщите следующий текст:',
            'ja': '以下の文章を要約してください:',
            'ko': '다음 텍스트를 요약하십시오:',
            'zh': '总结以下文本:',
            'ar': 'لخص النص التالي:',
            'hi': 'निम्नलिखित पाठ का सारांश दें:',
            'vi': 'Tóm tắt văn bản sau:',
            'th': 'สรุปข้อความต่อไปนี้:',
            'tr': 'Aşağıdaki metni özetle:',
            'pl': 'Podsumuj następujący tekst:'
        }

    def detect_language(self, text):
        try:
            lang = detect(text[:1000])  # Only use first 1000 chars for faster detection
            return lang if lang in self.language_codes else 'en'
        except:
            return 'en'

    def clean_and_format_points(self, summary_text, lang):
        # Language-specific sentence splitting patterns
        split_patterns = {
            'zh': r'[。！？]',
            'ja': r'[。！？]',
            'ko': r'[。.!?]',
            'th': r'[.!?]',
            'ar': r'[.!?؟।]',
            'hi': r'[.!?।]',
            'default': r'[.!?]'
        }
        
        # Choose appropriate split pattern
        split_pattern = split_patterns.get(lang, split_patterns['default'])
        
        # Split into sentences and clean up
        points = re.split(split_pattern, summary_text)
        points = [p.strip() for p in points if len(p.strip()) > 10]
        
        formatted_points = []
        seen = set()
        
        for point in points:
            # Normalize for deduplication
            normalized = re.sub(r'[^\w\s]', '', point.lower())
            if normalized in seen or len(normalized) < 10:
                continue
            seen.add(normalized)
            
            # Language-specific formatting
            if lang in ['ja', 'zh', 'ko']:
                # Remove spaces between CJK characters
                point = re.sub(r'(?<=[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af])\s+(?=[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af])', '', point)
                # Add proper CJK punctuation
                if not point.endswith(('。', '！', '？')):
                    point += '。'
            elif lang == 'ar':
                # Fix Arabic text direction and punctuation
                point = re.sub(r'\s+([.،؟!])', r'\1', point)
                if not point.endswith(('.', '؟', '!')):
                    point += '.'
            elif lang == 'th':
                # Thai specific cleaning
                point = re.sub(r'\s+([ๆ์])', r'\1', point)
                if not point.endswith(('.', '!')):
                    point += '.'
            elif lang == 'hi':
                # Hindi specific ending
                if not point.endswith(('.', '।', '!')):
                    point += '।'
            else:
                # Default punctuation
                if not point.endswith(('.', '!', '?')):
                    point += '.'
            
            formatted_points.append(f"• {point}")
        
        return formatted_points

    @torch.no_grad()
    def summarize(self, text, max_points=5):
        # Detect language
        lang = self.detect_language(text)
        mbart_lang = self.language_codes.get(lang, 'en_XX')
        
        # Clean and prepare text
        clean_text = text.strip().replace("\n", " ")
        prompt = self.language_prompts.get(lang, self.language_prompts['en'])
        input_text = f"{prompt}\n{clean_text}"
        
        # Set the source language
        self.tokenizer.src_lang = mbart_lang
        
        # Tokenize with appropriate padding
        inputs = self.tokenizer(input_text, 
                              return_tensors="pt",
                              max_length=1024,
                              truncation=True,
                              padding=True).to(self.device)
        
        # Generate summary
        self.tokenizer.tgt_lang = mbart_lang  # Set target language same as source
        summary_ids = self.model.generate(
            **inputs,
            decoder_start_token_id=self.tokenizer.lang_code_to_id[mbart_lang],
            max_length=200,
            min_length=30,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Format into bullet points
        bullet_points = self.clean_and_format_points(summary, lang)
        
        # Ensure we have enough points
        final_points = bullet_points[:max_points]
        
        return {
            "summary": "\n".join(final_points),
            "language": lang,
            "num_points": len(final_points),
            "model_lang": mbart_lang
        }

    def process_content(self, html_content):
        # First detect if it's an article
        is_article, text = self.article_detector.is_article(html_content)
        
        if not is_article:
            return {
                "is_article": False,
                "summary": None,
                "language": None,
                "num_points": 0
            }
        
        # If it is an article, proceed with summarization
        summary_result = self.summarize(text)
        summary_result["is_article"] = True
        
        return summary_result 