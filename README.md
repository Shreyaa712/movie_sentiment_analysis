# Movie Sentiment Analysis

A Python project that predicts the sentiment (Positive/Negative) of IMDb movie reviews using Natural Language Processing (NLP) and a Naive Bayes classifier.

## Features

- Text preprocessing (removes punctuation, numbers, converts to lowercase)
- TF-IDF vectorization for feature extraction
- Multinomial Naive Bayes classifier for sentiment prediction
- Evaluation using Accuracy, Confusion Matrix, and Classification Report
- Test custom movie reviews via user input

---

## How to Run

1. Download the **IMDBDataset.csv** file from Kaggle:  
   [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
2. Place the CSV **anywhere on your computer**.
3. Run `movie_sentiment.py`.
4. A file selector will appear → **select the CSV file**.
5. The program will train the model and show evaluation metrics.
6. Enter any movie review to predict its sentiment.

---

## Requirements

- Python
- Libraries:  
  pip install pandas scikit-learn
