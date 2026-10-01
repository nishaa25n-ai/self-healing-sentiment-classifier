# Self-Healing Sentiment Classification System

An end-to-end ML system that classifies Flipkart product reviews as
positive, negative, or neutral — and automatically detects when new
incoming data drifts from the training distribution, triggering
retraining without manual intervention or downtime.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Sentence Transformers, MLflow,
SciPy (KS-test), FastAPI, Docker

## How it works
1. Review text is converted to semantic embeddings using
   `all-MiniLM-L6-v2` (Sentence Transformers)
2. A Logistic Regression classifier predicts sentiment
   (`class_weight='balanced'` to handle class imbalance)
3. Every incoming request is buffered; a Kolmogorov-Smirnov test is
   run per embedding dimension against the training reference data
4. If a significant fraction of dimensions show drift AND enough
   labeled feedback has been collected, the model retrains on the
   combined old + new data
5. The new model is swapped into the live API with zero downtime
   (no restart needed)
6. All training runs are logged and versioned in MLflow

## API Endpoints
- `GET /status` — current model version, retrain count, buffered data
- `POST /predict` — classify a review's sentiment
- `POST /feedback` — submit the correct label for a prediction (used
  for retraining)
- `GET /debug_drift` — inspect current drift fraction

## Dataset
[Flipkart Product Reviews with Sentiment](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset)
(Kaggle), ~180k reviews. A 5,000-review sample was used for training.

## Results
- Accuracy: 91% (unbalanced) vs 77% (class-balanced)
- Balancing improved neutral-class recall from 0% → 42%
- Verified self-healing works end-to-end: fed the live API 280
  out-of-distribution requests with 60 labeled feedback examples —
  the system detected drift and retrained automatically
  (`model_version` incremented 1 → 2)

## Run locally with Docker
```bash
docker build -t sentiment-api .
docker run -p 8000:8000 sentiment-api
```
Then open `http://localhost:8000/docs` to try the API.

## Limitations
- Retraining needs labeled feedback (`/feedback`), which in a real
  deployment would come from users or manual review
- Trained on a 5k-review sample; training on the full dataset would
  likely improve accuracy
- In this container setup, the retrained model resets on restart —
  production would persist it to a volume or model registry# self-healing-sentiment-classifier
