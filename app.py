from flask import Flask, render_template, request
import joblib
import re
from nltk.corpus import stopwords

app = Flask(__name__)

model = joblib.load('models/logistic_regression_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/review-movie', methods=['GET', 'POST'])
def review_movie():
    prediction = None
    if request.method == 'POST':
        review = request.form['review']
        cleaned = preprocess(review)
        transformed = vectorizer.transform([cleaned])
        pred = model.predict(transformed)[0]
        prediction = 'Positive' if pred == 1 else 'Negative'
    return prediction

if __name__ == '__main__':
    app.run(debug=False)
