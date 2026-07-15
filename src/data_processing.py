"""
Data Processing Module
=======================
Handles loading data and preprocessing Arabic text.
"""
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
import torch
import os

# Ensure nltk resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ARABIC_STOPWORDS = set(stopwords.words('arabic'))

def preprocess_text(text: str) -> str:
    """
    Cleans and standardizes Arabic text.
    
    Steps:
    1. Removes English characters
    2. Removes "ال"
    3. Removes Arabic diacritics (الحركات)
    4. Removes punctuation
    5. Tokenizes
    6. Normalizes Alif, Hamza, and Taa Marboota
    7. Removes Stop words
    """
    if not isinstance(text, str):
        return ""
        
    # Remove English characters
    text = re.sub(r'[A-Za-z]', '', text)
    # Remove "ال"
    text = re.sub(r'\bال', '', text)
    # Remove Arabic diacritical marks (الحركات)
    diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(diacritics, '', text)
    # Remove punctuation and replace with space
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Convert همزات / تاء مربوطة
    tokens = [re.sub("[إأٱآا]", "ا", token) for token in tokens]
    tokens = [re.sub("ؤ", "ء", token) for token in tokens]
    tokens = [re.sub("ئ", "ء", token) for token in tokens]
    tokens = [re.sub("ة", "ه", token) for token in tokens]
    
    # Remove stop words
    tokens = [token for token in tokens if token not in ARABIC_STOPWORDS]
    
    # Join tokens back to text
    cleaned_text = ' '.join(tokens)
    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def load_and_preprocess_data(train_path: str, test_path: str):
    """
    Loads train and test excel files, cleans text, and encodes labels.
    
    Returns:
        X_train, X_test, y_train, y_test, label_encoder, train_df, test_df
    """
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Dataset files not found. Please ensure {train_path} and {test_path} exist."
        )

    print(f"Loading data from {train_path} and {test_path}...")
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)
    
    print("Preprocessing Arabic text (this may take a moment)...")
    train_df['News'] = train_df['News'].apply(preprocess_text)
    test_df['News'] = test_df['News'].apply(preprocess_text)
    
    X_train = train_df['News']
    X_test = test_df['News']
    
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(train_df['Type'])
    y_test_encoded = label_encoder.transform(test_df['Type'])
    
    # Return encoded labels as numpy arrays (easier for ML), 
    # PyTorch/Keras scripts can convert to tensors later.
    return X_train, X_test, y_train_encoded, y_test_encoded, label_encoder, train_df, test_df
