from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
import numpy as np
import string

df = pd.read_csv('Dataset/Processed dataset/processed_data.csv')

flat_array = df['sentence'].to_numpy().flatten()

flat_array_clean = np.array([
    str(text) if not pd.isna(text) else "" 
    for text in flat_array
])

vectorizer = TfidfVectorizer(max_features=2500, min_df=0.0, max_df=0.8)
X_vectorized = vectorizer.fit_transform(flat_array_clean)

classifier = joblib.load('Saved trained model/DecisionTree_model.joblib')

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))
stemmer = nltk.stem.SnowballStemmer('english')

label_dict = {4: 'sadness', 2: 'joy', 0: 'anger', 3: 'love', 5: 'surprise', 1: 'fear'}

def transform_data(sentence):
    sentence = sentence.lower()
    sentence = re.sub("^a-zA-Z0-9", ' ', sentence)
    sentence = re.sub('<.*?>', ' ', sentence)
    sentence = "".join([x for x in sentence if x not in string.punctuation])
    #sentence = stemmer.stem(sentence)
    sentence = sentence.split()
    sentence = [lemmatizer.lemmatize(x) for x in sentence if x not in stop_words]
    sentence = " ".join(sentence)
    return sentence

def predict_emotion(sentence):
    sentence = transform_data(sentence)

    flat_array = np.array(sentence).flatten()

    flat_array_clean = np.array([
        str(text) if not pd.isna(text) else "" 
        for text in flat_array
    ])

    sentence = vectorizer.transform(flat_array_clean)

    return label_dict[classifier.predict(sentence)[0]]

if __name__ == "__main__":
    test_sentence = input("Enter a sentence to predict its emotion: ")
    predicted_emotion = predict_emotion(test_sentence)
    print(f"Predicted emotion: {predicted_emotion}")
