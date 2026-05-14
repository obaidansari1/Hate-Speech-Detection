from flask import Flask, render_template, request, jsonify
import pickle
import re
from scipy.sparse import hstack

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    word_vec = pickle.load(f)
with open('char_vectorizer.pkl', 'rb') as f:
    char_vec = pickle.load(f)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\brt\b', '', text)
    text = re.sub(r"[^a-zA-Z\s']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyzer')
def analyzer():
    return render_template('analyzer.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_text = data.get('text', '')
    if not user_text:
        return jsonify({'error': 'No text'}), 400

    cleaned = clean_text(user_text)
    # Combined word + character n-gram features
    w = word_vec.transform([cleaned])
    c = char_vec.transform([cleaned])
    vectorized = hstack([w, c])

    prediction = model.predict(vectorized)[0]
    probs = model.predict_proba(vectorized)[0]

    mapping = {0: "Hate Speech", 1: "Offensive Language", 2: "Neutral Content"}
    scores = {
        'hate': round(probs[0] * 100, 1),
        'off':  round(probs[1] * 100, 1),
        'neu':  round(probs[2] * 100, 1)
    }

    return jsonify({'verdict': mapping.get(prediction, "Neutral Content"), 'scores': scores})

if __name__ == '__main__':
    app.run(debug=True)
