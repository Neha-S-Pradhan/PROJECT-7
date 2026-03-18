# =========================================
# DISASTER TWEET CLASSIFIER - TRAINING FILE
# =========================================

# --------- IMPORTS ---------
import os
import sys
import re
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from scipy.sparse import hstack

# --------- PATH SETUP ---------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(BASE_DIR, "data", "twitter_disaster.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Create models folder if not exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Add src to path for imports
sys.path.append(os.path.join(BASE_DIR, "src"))

from preprocess import clean_text


# --------- LOAD DATA ---------
print("📥 Loading dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded successfully")
print(df.head())


# --------- PREPROCESSING ---------
print("\n🧹 Cleaning text...")

df['clean_text'] = df['text'].astype(str).apply(clean_text)


# --------- FEATURE ENGINEERING ---------
print("\n⚙️ Creating additional features...")

df['tweet_length'] = df['text'].apply(len)
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
df['hashtag_count'] = df['text'].apply(lambda x: str(x).count('#'))
df['mention_count'] = df['text'].apply(lambda x: str(x).count('@'))


# --------- TF-IDF ---------
print("\n🔢 Vectorizing text...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9
)

X_text = vectorizer.fit_transform(df['clean_text'])


# --------- COMBINE FEATURES ---------
print("\n🔗 Combining features...")

extra_features = df[['tweet_length', 'word_count', 'hashtag_count', 'mention_count']].values

X = hstack((X_text, extra_features))
y = df['target']


# --------- TRAIN TEST SPLIT ---------
print("\n✂️ Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# --------- MODEL TRAINING ---------
print("\n🤖 Training model...")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("✅ Model training complete")


# --------- EVALUATION ---------
print("\n📊 Evaluating model...")

preds = model.predict(X_test)

accuracy = accuracy_score(y_test, preds)

print(f"\n✅ Accuracy: {accuracy:.4f}")
print("\n📄 Classification Report:\n", classification_report(y_test, preds))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, preds))


# --------- SAVE MODEL ---------
print("\n💾 Saving model...")

model_path = os.path.join(MODEL_DIR, "best_model.pkl")
vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(vectorizer_path, "wb") as f:
    pickle.dump(vectorizer, f)

print(f"✅ Model saved at: {model_path}")
print(f"✅ Vectorizer saved at: {vectorizer_path}")


# --------- DONE ---------
print("\n🎉 TRAINING COMPLETE SUCCESSFULLY!")