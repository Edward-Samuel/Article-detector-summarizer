from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

class TextSummarizer:
    def __init__(self):
        self.model_name = "t5-base"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def summarize(self, text, max_length=150, min_length=40):
        # Prepare the text input
        preprocess_text = text.strip().replace("\n", " ")
        t5_input = "summarize: " + preprocess_text

        # Tokenize the text
        tokenized_text = self.tokenizer.encode(t5_input, return_tensors="pt", max_length=512, truncation=True)
        tokenized_text = tokenized_text.to(self.device)

        # Generate Summary
        summary_ids = self.model.generate(
            tokenized_text,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary 