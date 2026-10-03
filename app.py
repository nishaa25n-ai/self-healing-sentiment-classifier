from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import joblib, threading
import numpy as np
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

app = FastAPI()
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
clf = joblib.load('sentiment_model.pkl')
X_ref = np.load('X_train_ref.npy')
y_ref = np.load('y_train_ref.npy', allow_pickle=True)

recent_X = []
fb_X, fb_y = [], []
state = {"model_version": 1, "retrain_count": 0}
lock = threading.Lock()

WINDOW = 200
MIN_FEEDBACK = 50

def drift_fraction(ref, new):
    drifted = sum(
        ks_2samp(ref[:, i], new[:, i]).pvalue < 0.05
        for i in range(ref.shape[1])
    )
    return drifted / ref.shape[1]

def drift_detected(ref, new, frac_threshold=0.15):
    return drift_fraction(ref, new) > frac_threshold

def maybe_heal():
    global clf, X_ref, y_ref
    if len(recent_X) < WINDOW or len(fb_X) < MIN_FEEDBACK:
        return
    new = np.array(recent_X[-WINDOW:])
    drifted = drift_detected(X_ref, new)

    if not drifted:
        recent_X.clear()   # drift nahi mila, buffer clear karo taaki purana data repeat na ho
        return

    with lock:
        X = np.vstack([X_ref, np.array(fb_X)])
        y = np.concatenate([y_ref, np.array(fb_y)])
        new_clf = LogisticRegression(max_iter=1000, class_weight='balanced').fit(X, y)
        joblib.dump(new_clf, 'sentiment_model.pkl')
        clf = new_clf
        X_ref, y_ref = X, y
        recent_X.clear(); fb_X.clear(); fb_y.clear()
        state["model_version"] += 1
        state["retrain_count"] += 1

class ReviewInput(BaseModel):
    text: str

class Feedback(BaseModel):
    text: str
    label: str

@app.get("/")
def home():
    return {"message": "Sentiment Classification API is running!"}

@app.get("/status")
def status():
    return {**state, "buffered_requests": len(recent_X), "feedback_samples": len(fb_X)}

@app.get("/debug_drift")
def debug_drift():
    if len(recent_X) < 10:
        return {"error": "not enough buffered requests yet"}
    new = np.array(recent_X[-min(len(recent_X), 200):])
    frac = drift_fraction(X_ref, new)
    return {"drift_fraction": frac, "buffered": len(recent_X)}

@app.post("/predict")
def predict(review: ReviewInput, bg: BackgroundTasks):
    emb = embed_model.encode([review.text])
    recent_X.append(emb[0])
    pred = clf.predict(emb)[0]
    if len(recent_X) % 50 == 0:
        bg.add_task(maybe_heal)
    return {"review": review.text, "sentiment": pred, "model_version": state["model_version"]}

@app.post("/feedback")
def feedback(fb: Feedback):
    emb = embed_model.encode([fb.text])
    fb_X.append(emb[0]); fb_y.append(fb.label)
    return {"feedback_samples": len(fb_X)}
