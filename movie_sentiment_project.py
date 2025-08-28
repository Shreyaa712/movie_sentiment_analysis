# Movie Sentiment Analysis - IDLE Friendly (CSV Selection)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import re
import string
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Step 1: Ask user to select the CSV file
Tk().withdraw()  # Hide the small Tkinter window
file_path = askopenfilename(title="Select IMDBDataset.csv")
if not file_path:
    print("No file selected. Exiting...")
    exit()

# Step 2: Load the dataset
data = pd.read_csv(file_path)
print("Dataset loaded successfully!")
print("First 5 rows:\n", data.head())

# Step 3: Inspect the data
print("\nColumns:", data.columns)
print("\nMissing values:\n", data.isnull().sum())
print("\nValue counts for sentiment:\n", data['sentiment'].value_counts())

# Step 4: Clean the text
def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()  # Lowercase
    return text

print("\nCleaning text... This may take a few seconds.")
data['clean_review'] = data['review'].apply(clean_text)
print("Text cleaned!")

# Step 5: Split data
X = data['clean_review']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("\nData split into training and testing sets.")

# Step 6: Vectorize text using TF-IDF
print("\nVectorizing text using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("Vectorization done!")

# Step 7: Train Naive Bayes classifier
print("\nTraining Naive Bayes classifier...")
model = MultinomialNB()
model.fit(X_train_vec, y_train)
print("Model trained!")

# Step 8: Make predictions
y_pred = model.predict(X_test_vec)

# Step 9: Evaluate the model
print("\n--- Model Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Test your own review
sample_review = input("\nEnter a movie review to predict sentiment: ")
sample_review_clean = clean_text(sample_review)
sample_vec = vectorizer.transform([sample_review_clean])
prediction = model.predict(sample_vec)
print("Predicted sentiment:", prediction[0])

