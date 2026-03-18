# =========================================
# DISASTER TWEET CLASSIFIER - FLASK APP
# (UPDATED FOR FEATURE ENGINEERING MODEL)
# =========================================

from flask import Flask, render_template, request
import pickle
import os
import sys
import numpy as np

# --------- PATH SETUP ---------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_PATH = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_PATH)

from preprocess import clean_text

# --------- LOAD MODEL ---------
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

if not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(f"Vectorizer not found at {VECTORIZER_PATH}")

print("🔄 Loading model...")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

print("✅ Model loaded successfully!")

# --------- FLASK APP ---------
app = Flask(__name__)

# --------- FEATURE FUNCTION ---------
def extract_extra_features(text):
    """
    Generate same features used during training
    """
    tweet_length = len(text)
    word_count = len(text.split())
    hashtag_count = text.count('#')
    mention_count = text.count('@')

    return np.array([[tweet_length, word_count, hashtag_count, mention_count]])

# --------- PREDICTION ---------
def predict_tweet(text):
    cleaned = clean_text(text)

    # TF-IDF
    X_text = vectorizer.transform([cleaned])

    # Extra features
    extra = extract_extra_features(text)

    # Combine
    from scipy.sparse import hstack
    X = hstack((X_text, extra))

    prediction = model.predict(X)[0]

    return "🚨 Disaster Tweet" if prediction == 1 else "✅ Non-Disaster Tweet"

# --------- ROUTES ---------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("tweet")

        if user_text and user_text.strip():
            prediction = predict_tweet(user_text)
        else:
            prediction = "⚠️ Please enter some text"

    return render_template("index.html", prediction=prediction, user_text=user_text)

# --------- RUN ---------
if __name__ == "__main__":
    print("🚀 Starting Flask app...")
    app.run(debug=True)