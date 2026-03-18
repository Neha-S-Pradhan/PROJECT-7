# =========================================
# IMPORTS
# =========================================
import os
import sys
import pickle

# Fix import path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from preprocess import clean_text


# =========================================
# PATH SETUP
# =========================================
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")


# =========================================
# LOAD MODEL
# =========================================
print("Loading model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

if not os.path.exists(VECTORIZER_PATH):
    raise FileNotFoundError(f"Vectorizer not found at: {VECTORIZER_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

print("Model loaded successfully!")


# =========================================
# PREDICTION FUNCTION
# =========================================
def predict_tweet(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    if prediction == 1:
        return "🚨 Disaster Tweet"
    else:
        return "✅ Non-Disaster Tweet"


# =========================================
# USER INPUT LOOP
# =========================================
print("\nEnter tweet to classify (type 'exit' to quit)\n")

while True:
    user_input = input("Tweet: ")

    if user_input.lower() == "exit":
        print("Exiting...")
        break

    result = predict_tweet(user_input)
    print("Prediction:", result)
    print("-" * 50)