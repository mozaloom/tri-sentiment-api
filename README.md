# Twitter Sentiment Analysis API

This project provides a FastAPI-based web service for Twitter sentiment analysis using three different models:

* **RoBERTa** (transformer-based)
* **TF-IDF + Passive Aggressive Classifier (PAC)**
* **Bag-of-Words (BoW) + PAC**

## Project Structure

```
.
├── data/          # Training & validation CSVs
├── models/        # Saved vectorizers & classifiers
├── notebooks/     # Model development & training notebook
├── webapp/        # FastAPI application
├── Dockerfile     # Container setup
├── Makefile       # Automation tasks
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the API**

   ```bash
   uvicorn webapp.app:app --host 0.0.0.0 --port 8000
   ```

3. **Using Docker (optional)**

   ```bash
   docker build -t sentiment-api .
   docker run -p 8000:8000 sentiment-api
   ```

## API Endpoints

* `POST /predict/roberta` – RoBERTa-based inference
* `POST /predict/tfidf`   – TF-IDF + PAC model
* `POST /predict/bow`     – BoW + PAC model

**Example Request**

```json
{ "text": "I love this product!" }
```

**Example Response**

```json
{ "label": "positive" }
```

## Model Training

All training steps are in `notebooks/notebook-sa.ipynb`. In summary:

1. **Data Loading**
   Read `data/twitter_training.csv` and `data/twitter_validation.csv` (columns: `text`, `label`).

2. **Text Preprocessing**

   * Lowercasing, tokenization, stopword removal (using NLTK).
   * Optional lemmatization/stemming.

3. **TF-IDF & BoW Vectorizers**

   * **TF-IDF**: `TfidfVectorizer(max_features=5000)`
   * **BoW**:   `CountVectorizer(max_features=5000)`

4. **Classifier Training**
   For each vectorizer, train a `PassiveAggressiveClassifier`:

   ```python
   from sklearn.linear_model import PassiveAggressiveClassifier
   clf = PassiveAggressiveClassifier(max_iter=1000)
   clf.fit(X_train, y_train)
   ```

   Validate on the hold-out set and save with `joblib.dump()` into `models/`.

5. **RoBERTa Inference**
   Leverage the pretrained `cardiffnlp/twitter-roberta-base-sentiment` model from Hugging Face for zero-shot sentiment inference. (No fine-tuning in the notebook, but you can add it by following the `transformers` fine-tuning tutorial.)

6. **Saving Artifacts**

   * `models/tfidf_vectorizer.pkl`
   * `models/bow_vectorizer.pkl`
   * `models/pac_tfidf.pkl`
   * `models/pac_bow.pkl`

Re-run all cells in the notebook to reproduce or adjust hyperparameters, then re-save into `models/`.

## License

This project is licensed under the [MIT License](./LICENSE).
