from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import string
import os

app = Flask(__name__)

# -------------------------------------------
# Ensure NLTK resources exist (Render safe)
# -------------------------------------------

nltk.data.path.append('./nltk_data')

# Download punkt
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir="./nltk_data")

# Download punkt_tab
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", download_dir="./nltk_data")

# Download stopwords
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir="./nltk_data")


# -------------------------------------------
# Text preprocessing function
# -------------------------------------------
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    sw = set(stopwords.words('english'))

    for i in text:
        if i not in sw and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# -------------------------------------------
# Prediction function
# -------------------------------------------
def predict_spam(message):
    transformed_sms = transform_text(message)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    return result


# -------------------------------------------
# Routes
# -------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    result = predict_spam(message)
    return render_template('index.html', result=result)


# -------------------------------------------
# Load ML model + Vectorizer
# -------------------------------------------
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))


# -------------------------------------------
# Start App (Render)
# -------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0')
