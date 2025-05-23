import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
import joblib
from pathlib import Path
import os

def train_and_save_models():
    try:
        # Base directories
        BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        MODEL_DIR = BASE_DIR / 'Saved trained model'
        DATASET_DIR = BASE_DIR / 'Dataset'

        # Ensure the model directory exists
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        print("Loading dataset...")
        df = pd.read_csv(DATASET_DIR / 'Processed dataset' / 'processed_data.csv')

        # Remove rows with NaN values and ensure data types
        df = df.dropna(subset=['sentence', 'label'])
        X = df['sentence'].astype(str).values
        y = df['label'].values

        print(f"Dataset shape after cleaning: {df.shape}")

        # Initialize and fit vectorizer
        print("Initializing vectorizer...")
        vectorizer = TfidfVectorizer(max_features=2500, min_df=0.0, max_df=0.8)
        X_vectorized = vectorizer.fit_transform(X)

        # Save the vectorizer
        print("Saving vectorizer...")
        joblib.dump(vectorizer, MODEL_DIR / 'vectorizer.joblib')

        # Train and save models
        print("Training and saving models...")

        # Logistic Regression
        print("Training Logistic Regression...")
        logistic_model = LogisticRegression(max_iter=1000, random_state=42)
        logistic_model.fit(X_vectorized, y)
        joblib.dump(logistic_model, MODEL_DIR / 'LogisticRegression_model.joblib')

        # Decision Tree
        print("Training Decision Tree...")
        decision_tree_model = DecisionTreeClassifier(random_state=42)
        decision_tree_model.fit(X_vectorized, y)
        joblib.dump(decision_tree_model, MODEL_DIR / 'DecisionTree_model.joblib')

        # Linear SVC
        print("Training Linear SVC...")
        linear_svc_model = LinearSVC(random_state=42)
        linear_svc_model.fit(X_vectorized, y)
        joblib.dump(linear_svc_model, MODEL_DIR / 'LinearSVC_model.joblib')

        # Multinomial Naive Bayes
        print("Training Multinomial Naive Bayes...")
        multinomial_nb_model = MultinomialNB()
        multinomial_nb_model.fit(X_vectorized, y)
        joblib.dump(multinomial_nb_model, MODEL_DIR / 'MultinomialNB_model.joblib')

        print("All models have been retrained and saved successfully!")
        return True

    except Exception as e:
        print(f"Error during model training: {str(e)}")
        return False

if __name__ == "__main__":
    success = train_and_save_models()
    if not success:
        exit(1) 