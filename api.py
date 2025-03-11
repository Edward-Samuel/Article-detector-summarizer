from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from article_classifier import ArticleClassifier
from text_summarizer import TextSummarizer
import uvicorn

app = FastAPI()

# Initialize models
article_classifier = ArticleClassifier()
text_summarizer = TextSummarizer()

class WebContent(BaseModel):
    html_content: str

@app.post("/process")
async def process_content(content: WebContent):
    try:
        # Check if it's an article and extract content
        is_article, main_content, detected_language = article_classifier.predict(content.html_content)
        
        if not is_article:
            return {
                "is_article": False,
                "content": None,
                "summary": None,
                "language": None
            }
            
        # Generate summary
        summary_result = text_summarizer.summarize(main_content)
        
        return {
            "is_article": True,
            "content": main_content,
            "summary": summary_result["summary"],
            "language": detected_language
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080) 