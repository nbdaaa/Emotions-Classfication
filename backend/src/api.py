import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import tiktoken
import sklearn
from pathlib import Path

from Model.Transformer.TransformerEncoder import TransformerModel
from Model.TransformerWithROPE.TransformerEncoderROPE import TransformerModelWithROPE 

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pandas as pd
import numpy as np
import string

print("Python:", sys.version)
print("joblib:", joblib.__version__)
print("scikit-learn:", sklearn.__version__)

# Add root directory to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent  # /app
sys.path.append(str(ROOT_DIR / 'src')) 

# Base directories
BASE_DIR = ROOT_DIR
MODEL_DIR = BASE_DIR / 'Saved trained model'
DATASET_DIR = BASE_DIR / 'Dataset'

# Flask app init
app = Flask(__name__)
CORS(app)

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize models and components
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
tokenizer = tiktoken.get_encoding('gpt2')

# Load traditional ML models
logistic_model = joblib.load(MODEL_DIR / 'LogisticRegression_model.joblib')
decision_tree_model = joblib.load(MODEL_DIR / 'DecisionTree_model.joblib')
linear_svc_model = joblib.load(MODEL_DIR / 'LinearSVC_model.joblib')
multinomial_nb_model = joblib.load(MODEL_DIR / 'MultinomialNB_model.joblib')

# Initialize vectorizer
df = pd.read_csv(DATASET_DIR / 'Processed dataset' / 'processed_data.csv')
flat_array_clean = np.array([str(text) if not pd.isna(text) else "" for text in df['sentence'].to_numpy().flatten()])
vectorizer = TfidfVectorizer(max_features=2500, min_df=0.0, max_df=0.8)
vectorizer.fit_transform(flat_array_clean)

# Initialize transformer models
vocab_size = tokenizer.n_vocab
embed_size = 256
d_model = 256
num_heads = 8
d_ff = 512
output_size = 6
num_layers = 3
dropout = 0.2

# Load transformer model with CPU mapping
transformer_model = TransformerModel(vocab_size, embed_size, d_model, num_heads, d_ff, output_size, num_layers, dropout)
transformer_model.load_state_dict(
    torch.load(MODEL_DIR / 'best_transformer_model.pth', map_location=device)
)
transformer_model = transformer_model.to(device)
transformer_model.eval()

# Load transformer ROPE model with CPU mapping
transformer_rope_model = TransformerModelWithROPE(vocab_size, embed_size, d_model, num_heads, d_ff, output_size, num_layers, dropout)
transformer_rope_model.load_state_dict(
    torch.load(MODEL_DIR / 'best_transformerROPE_model.pth', map_location=device)
)
transformer_rope_model = transformer_rope_model.to(device)
transformer_rope_model.eval()

# Initialize text processing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
label_dict = {4: 'sadness', 2: 'joy', 0: 'anger', 3: 'love', 5: 'surprise', 1: 'fear'}

def transform_data(sentence):
    sentence = sentence.lower()
    sentence = re.sub("^a-zA-Z0-9", ' ', sentence)
    sentence = re.sub('<.*?>', ' ', sentence)
    sentence = "".join([x for x in sentence if x not in string.punctuation])
    sentence = sentence.split()
    sentence = [lemmatizer.lemmatize(x) for x in sentence if x not in stop_words]
    return " ".join(sentence)

def predict_traditional_ml(sentence, model):
    sentence = transform_data(sentence)
    sentence_vectorized = vectorizer.transform([sentence])
    return label_dict[model.predict(sentence_vectorized)[0]]

def predict_transformer(sentence, model):
    sentence = sentence.lower()
    encoding = tokenizer.encode(sentence)
    input_ids = torch.tensor(encoding, dtype=torch.long, device=device).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_ids)
        _, predicted = torch.max(outputs, 1)
    
    return label_dict[predicted.item()]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sentence = data.get('sentence', '')
    model_type = data.get('model', 'logistic')
    
    print(f"Received sentence: {sentence}")
    print(f"Received model type: {model_type}")
    
    if not sentence:
        return jsonify({'error': 'No sentence provided'}), 400
    
    try:
        if model_type == 'logistic':
            prediction = predict_traditional_ml(sentence, logistic_model)
        elif model_type == 'decision_tree':
            prediction = predict_traditional_ml(sentence, decision_tree_model)
        elif model_type == 'linear_svc':
            prediction = predict_traditional_ml(sentence, linear_svc_model)
        elif model_type == 'multinomial_nb':
            prediction = predict_traditional_ml(sentence, multinomial_nb_model)
        elif model_type == 'transformer':
            prediction = predict_transformer(sentence, transformer_model)
        elif model_type == 'transformer_rope':
            prediction = predict_transformer(sentence, transformer_rope_model)
        else:
            return jsonify({'error': 'Invalid model type'}), 400
        
        return jsonify({'emotion': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
