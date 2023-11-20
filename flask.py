# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:11:32 2023

@author: jaswa
"""

from flask import Flask, request
import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import pickle
from flask_cors import CORS
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from lstm import SentimentRNN
import numpy as np
import re

# Initializing flask app
app = Flask(__name__)
app.debug = True
CORS(app)
 
# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
device = torch.device("cpu")

with open('C:/Users/jaswa/OneDrive/Desktop/SWM/preprocessing_models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
with open('C:/Users/jaswa/OneDrive/Desktop/SWM/models/logistic_regressioin.pkl', 'rb') as f:
    logistic_regressioin = pickle.load(f)
with open('C:/Users/jaswa/OneDrive/Desktop/SWM/models/RandomForest.pkl', 'rb') as f:
    randomforest = pickle.load(f)
with open('C:/Users/jaswa/OneDrive/Desktop/SWM/models/svc.pkl', 'rb') as f:
    svc = pickle.load(f)


with open('C:/Users/jaswa/OneDrive/Desktop/SWM/preprocessing_models/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
no_layers = 2
vocab_size = 1001 
embedding_dim = 64
output_dim = 1
hidden_dim = 256
lstm = SentimentRNN(no_layers,vocab_size,hidden_dim,embedding_dim,drop_prob=0.5)
lstm.load_state_dict(torch.load('C:/Users/jaswa/OneDrive/Desktop/SWM/models/lstm_cpu.pth',map_location=torch.device('cpu')))
lstm.eval()
    
bert_model_name = 'bert-base-uncased'

# Tokenizer
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertForSequenceClassification.from_pretrained("C:/Users/jaswa/OneDrive/Desktop/SWM/models/bert_model/")


# Function to preprocess text
def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text)

    tokens = [w.lower() for w in tokens]

    # Remove punctuation from each token
    words = [word for word in tokens if word.isalpha()]

    # Filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]

    # Stemming
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]

    return lemmatized



def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

def randomforest_predict(text):
    processed_text = preprocess_text(text)
    vector = tfidf_vectorizer.transform([text])
    return randomforest.predict(vector)[0]


def logisticregression_predict(text):
    processed_text = preprocess_text(text)
    vector = tfidf_vectorizer.transform([text])
    return logistic_regressioin.predict(vector)[0]

def svc_predict(text):
    processed_text = preprocess_text(text)
    vector = tfidf_vectorizer.transform([text])
    return svc.predict(vector)[0]
    

def lstm_predict(text):
        word_seq = np.array([vocab[preprocess_string(word)] for word in text.split()
                if preprocess_string(word) in vocab.keys()])
        word_seq = np.expand_dims(word_seq,axis=0)
        pad =  torch.from_numpy(padding_(word_seq,500))
        inputs = pad.to(device)
        batch_size = 1
        h = lstm.init_hidden(batch_size)
        h = tuple([each.data for each in h])
        output, h = lstm(inputs, h)
        return(output.item())
    
def bert_predict(text):
    tokens = bert_tokenizer(text, return_tensors='pt')

# Forward pass through the model
    with torch.no_grad():
        outputs = bert_model(**tokens)

# Get the predicted logits
    logits = outputs.logits

# Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    _, preds = torch.max(logits, dim=1)
    return preds.numpy()[0]
    
    
@app.route('/predict-sentiment', methods=['GET'])
def predict():
    model = request.headers.get('model')
    text = request.headers.get('text')
    if model == "LR":
        v = logisticregression_predict(text)
    elif model == "RF":
        v = randomforest_predict(text)
    elif model == "LSTM":
        v = lstm_predict(text)
    elif model == "BERT":
        v = bert_predict(text)
    elif model == "SVC":
        v = svc_predict(text)
    
    if v >= 0.5:
        res = "positive"
    else:
        res = "negative"
    return jsonify(res)
    
    
    
     
# Running app
if __name__ == '__main__':
    app.run(debug=True)
    pass
