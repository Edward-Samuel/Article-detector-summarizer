import torch
from transformers import BertTokenizer, BertForSequenceClassification
from bs4 import BeautifulSoup
import re

class ArticleClassifier:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def preprocess_text(self, html_content):
        # Parse HTML and extract text
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
            
        # Get text content
        text = soup.get_text()
        
        # Clean text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
        
    def extract_main_content(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['header', 'nav', 'footer', 'aside', 'script', 'style', 'iframe', 'form']):
            element.decompose()
            
        # Look for common article containers
        article_containers = soup.find_all(['article', 'main']) or \
                           soup.find_all('div', class_=lambda x: x and any(term in str(x).lower() 
                                       for term in ['article', 'content', 'post', 'story', 'main']))
        
        if article_containers:
            main_content = max(article_containers, key=lambda x: len(x.get_text()))
            return self._clean_content(main_content.get_text())
            
        # Fallback to finding the longest content div
        main_content = None
        max_length = 0
        min_text_length = 500  # Minimum text length for article consideration
        
        for div in soup.find_all('div'):
            text = div.get_text()
            text_length = len(text)
            
            # Check for article indicators
            has_paragraphs = len(div.find_all('p')) > 2
            has_text_density = text_length / (len(str(div)) + 1) > 0.5
            
            if text_length > max_length and has_paragraphs and has_text_density:
                max_length = text_length
                main_content = div
                
        if main_content and max_length > min_text_length:
            return self._clean_content(main_content.get_text())
        return ""
    
    def _clean_content(self, text):
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove multiple newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
        
    def predict(self, html_content):
        # Check for article indicators in HTML structure
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Strong indicators of an article
        article_indicators = {
            'has_article_tag': bool(soup.find('article')),
            'has_paragraphs': len(soup.find_all('p')) > 3,
            'has_headings': bool(soup.find(['h1', 'h2', 'h3'])),
            'has_article_class': bool(soup.find(class_=lambda x: x and 'article' in str(x).lower())),
            'has_content_length': len(soup.get_text()) > 1000
        }
        
        # Extract main content first
        main_content = self.extract_main_content(html_content)
        
        # If we have strong structural indicators and substantial content, it's likely an article
        if sum(article_indicators.values()) >= 3 and len(main_content) > 500:
            return True, main_content
            
        # Fallback to BERT classification for ambiguous cases
        text = self.preprocess_text(html_content)
        
        # Tokenize
        inputs = self.tokenizer(text[:512], 
                              return_tensors="pt",
                              truncation=True,
                              padding=True)
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.softmax(outputs.logits, dim=1)
            
        is_article = predictions[0][1].item() > 0.5
        
        if is_article and main_content:
            return True, main_content
        
        return False, ""

    def train(self, train_data):
        """
        Train the model with labeled data
        train_data: List of tuples (html_content, is_article)
        """
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        for html_content, label in train_data:
            text = self.preprocess_text(html_content)
            inputs = self.tokenizer(text[:512], 
                                  return_tensors="pt",
                                  truncation=True,
                                  padding=True)
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            labels = torch.tensor([label]).to(self.device)
            
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() 