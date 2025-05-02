from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax

app = FastAPI(
    title="Sentiment Analysis API",
    description="API offering sentiment predictions using three different models: RoBERTa, TF-IDF + PassiveAggressive, and Bag-of-Words + PassiveAggressive.",
    version="1.0.0"
)

class TextRequest(BaseModel):
    text: str

# Load TF-IDF model and vectorizer
try:
    tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    tfidf_model = joblib.load("models/pac_classifier.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load TF-IDF model: {e}")

# Load Bag-of-Words model and vectorizer
try:
    with open("models/bow_vectorizer.pkl", "rb") as f:
        bow_vectorizer = pickle.load(f)
    with open("models/pac_sentiment_model.pkl", "rb") as f:
        bow_model = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load BOW model: {e}")

# Load RoBERTa model and tokenizer
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    roberta_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
except Exception as e:
    raise RuntimeError(f"Failed to load RoBERTa model: {e}")


def get_roberta_sentiment(text: str):
    """Return sentiment probabilities and label using RoBERTa"""
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = roberta_model(**encoded)
    scores = softmax(output.logits[0].detach().numpy())
    labels = ["negative", "neutral", "positive"]
    result = {f"roberta_{lbl}": float(scores[i]) for i, lbl in enumerate(labels)}
    result["label"] = labels[int(scores.argmax())]
    return result


def get_classifier_sentiment(text: str, vectorizer, model):
    """Return sentiment label using a scikit-learn classifier"""
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    label = "positive" if pred == 1 else "negative"
    return {"label": label}


@app.post("/predict/roberta")
def predict_roberta(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")
    return get_roberta_sentiment(req.text)


@app.post("/predict/tfidf")
def predict_tfidf(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")
    return get_classifier_sentiment(req.text, tfidf_vectorizer, tfidf_model)


@app.post("/predict/bow")
def predict_bow(req: TextRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")
    return get_classifier_sentiment(req.text, bow_vectorizer, bow_model)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
