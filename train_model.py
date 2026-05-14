import pandas as pd
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("clean_dataset.csv")
print("Dataset loaded. Shape:", data.shape)
print("Class distribution:")
print(data["class"].value_counts())

# ────────────────────────────────────────────────────
# Improved preprocessing — NO stopword removal.
# Removing stopwords destroys context:
#   "you are disgusting" → "disgusting" (lost sentiment structure)
# Keeping the full sentence gives the model far more signal.
# ────────────────────────────────────────────────────
def smart_clean(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)   # remove URLs
    text = re.sub(r'@\w+', '', text)              # remove @mentions
    text = re.sub(r'\brt\b', '', text)            # remove RT prefix
    text = re.sub(r"[^a-zA-Z\s']", ' ', text)    # keep letters + apostrophes
    text = re.sub(r'\s+', ' ', text).strip()
    return text

data['smart_text'] = data['tweet'].apply(smart_clean)

# Balance dataset — upsample minority classes to match majority (offensive)
df0 = data[data["class"] == 0]   # hate speech (1430)
df1 = data[data["class"] == 1]   # offensive  (19190)
df2 = data[data["class"] == 2]   # neutral    (4163)

df0_up = resample(df0, replace=True, n_samples=len(df1), random_state=42)
df2_up = resample(df2, replace=True, n_samples=len(df1), random_state=42)

balanced = pd.concat([df0_up, df1, df2_up])
print("\nBalanced distribution:")
print(balanced["class"].value_counts())

X = balanced["smart_text"]
y = balanced["class"]

# Vectorizer: word n-grams (1–3), sublinear TF scaling, larger vocab
vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 3),
    min_df=1,
    sublinear_tf=True,
)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression — reliable and fast for text classification
model = LogisticRegression(
    max_iter=1000,
    C=5.0,
    class_weight='balanced',
    solver='lbfgs'
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, preds):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=["Hate Speech", "Offensive", "Neutral"]))

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("\n✅ model.pkl and vectorizer.pkl saved successfully!")
