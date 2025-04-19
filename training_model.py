import pandas as pd
import numpy as np
import re # Regular expressions for text cleaning
import nltk # Natural Language Toolkit
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer # Optional: for stemming
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # To save/load models and vectorizer (optional)

# 1.2 Load Data
# Assuming the CSV file is named 'IMDB Dataset.csv' and is in the same directory or accessible path
try:
    df = pd.read_csv('IMDB Dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'IMDB Dataset.csv' not found. Please download it from Kaggle and place it in the correct directory.")
    # You might want to exit or handle this error more robustly
    exit() # Simple exit if file not found

# Display basic info and first few rows
print("\nDataset Info:")
df.info()

print("\nFirst 5 rows:")
print(df.head())

# Map sentiment labels to numerical values
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

print("\nSentiment value counts (1: positive, 0: negative):")
print(df['sentiment'].value_counts())

# 2. Text Preprocessing Function
stop_words = set(stopwords.words('english'))
# ps = PorterStemmer() # Initialize stemmer if using

def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and numbers, keep only letters
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    # Convert to lowercase
    text = text.lower()
    # Tokenize (split into words) and remove stop words
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Optional: Stemming
    # words = [ps.stem(word) for word in words]
    # Join words back into a string
    text = ' '.join(words)
    return text

# Apply the preprocessing function to the 'review' column
print("\nPreprocessing text data... (This may take a few minutes)")
# Create a new column for cleaned reviews
df['cleaned_review'] = df['review'].apply(preprocess_text)
print("Text preprocessing complete.")

# Display original vs cleaned review for one example
print("\nExample Preprocessing:")
print("Original:", df['review'][0][:200] + "...") # Show first 200 chars
print("Cleaned:", df['cleaned_review'][0][:200] + "...")

# 3.1 Split Data into Training and Testing Sets
X = df['cleaned_review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
# stratify=y ensures the proportion of positive/negative reviews is similar in train and test sets

print(f"\nData Split:")
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# 3.2 TF-IDF Vectorization
# Initialize TF-IDF Vectorizer
# max_features limits the vocabulary size to the most frequent terms, useful for large datasets
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # You can tune max_features

# Fit the vectorizer on the training data and transform the training data
print("\nFitting TF-IDF Vectorizer and transforming training data...")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the test data using the *same* fitted vectorizer
print("Transforming test data...")
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print("TF-IDF transformation complete.")
print(f"Shape of TF-IDF matrix (Train): {X_train_tfidf.shape}") # (num_samples, num_features)
print(f"Shape of TF-IDF matrix (Test): {X_test_tfidf.shape}")

# 4. Model Training
# Initialize Logistic Regression model
# C is the inverse of regularization strength; smaller C means stronger regularization.
# max_iter might need adjustment for convergence depending on the data/solver.
log_reg = LogisticRegression(C=1.0, max_iter=1000, random_state=42, solver='liblinear') # liblinear is good for binary classification with larger datasets

print("\nTraining Logistic Regression model...")
log_reg.fit(X_train_tfidf, y_train)
print("Model training complete.")

# Optional: Save the model and vectorizer
joblib.dump(log_reg, 'logistic_regression_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
print("Model and Vectorizer saved.")

# 5. Model Evaluation
# Make predictions on the test set
print("\nEvaluating model on the test set...")
y_pred = log_reg.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Format:
# [[TN, FP],
#  [FN, TP]]

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)']))