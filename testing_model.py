import joblib
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
log_reg = joblib.load('logistic_regression_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

new_reviews = [
    "This movie was absolutely fantastic! The acting was superb and the storyline kept me engaged throughout.",
    "What a waste of time. The plot was predictable and the characters were incredibly boring. I would not recommend this film.",
    "It was an okay movie, not great but not terrible either. Some good moments but overall quite average."
]

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    return text

def review(reviews):
    cleaned_new_reviews = [preprocess_text(review) for review in reviews]
    new_reviews_tfidf = tfidf_vectorizer.transform(cleaned_new_reviews)
    new_predictions = log_reg.predict(new_reviews_tfidf)
    sentiment_labels = {1: 'Positive', 0: 'Negative'}

    for review, prediction in zip(reviews, new_predictions):
        print(f"\nReview: \"{review[:100]}...\"")
        print(f"Predicted Sentiment: {sentiment_labels[prediction]} ({prediction})")

reviews = []
review_text = input("What is your review?: ")
reviews.append(review_text)
review(reviews)