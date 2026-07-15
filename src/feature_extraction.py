"""
Feature Extraction Module
==========================
Generates numerical representations for text using:
1. TF-IDF
2. CountVectorizer (One-Hot)
3. Word2Vec (Skip-gram and CBoW)
4. BERT Embeddings
"""

import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Conditionally import heavy libraries
try:
    import gensim
except ImportError:
    gensim = None

try:
    from transformers import BertTokenizer, BertModel
    import torch
except ImportError:
    torch = None


def extract_tfidf(X_train, X_test, max_features=300):
    """Extract TF-IDF features."""
    print(f"Extracting TF-IDF features (max_features={max_features})...")
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    X_train_df = pd.DataFrame(X_train_vec.toarray(), columns=vectorizer.get_feature_names_out())
    X_test_df = pd.DataFrame(X_test_vec.toarray(), columns=vectorizer.get_feature_names_out())
    
    return X_train_df, X_test_df, vectorizer

def extract_count_vectorizer(X_train, X_test, max_features=300):
    """Extract Binary CountVectorizer (One-Hot) features."""
    print(f"Extracting One-Hot features (max_features={max_features})...")
    vectorizer = CountVectorizer(binary=True, max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    X_train_df = pd.DataFrame(X_train_vec.toarray(), columns=vectorizer.get_feature_names_out())
    X_test_df = pd.DataFrame(X_test_vec.toarray(), columns=vectorizer.get_feature_names_out())
    
    return X_train_df, X_test_df, vectorizer


def get_doc_vec(sent, model):
    """Average word vectors for a document."""
    w2v_embeddings = []
    if not isinstance(sent, str):
        return None
        
    tokens = sent.split()
    for word in tokens:
        try:
            w2v_embeddings.append(model.wv[word])
        except KeyError:
            continue
            
    if len(w2v_embeddings) == 0:
        return None
    return np.mean(w2v_embeddings, axis=0)


def extract_word2vec(X_train, X_test, model_path: str):
    """
    Extract features using a pre-trained Word2Vec model.
    """
    if gensim is None:
        raise ImportError("Please install gensim to use Word2Vec.")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Word2Vec model not found at {model_path}. Please place it there.")
        
    print(f"Loading Word2Vec model from {model_path}...")
    model = gensim.models.Word2Vec.load(model_path)
    
    print("Generating embeddings...")
    X_train_emb = X_train.apply(lambda sent: get_doc_vec(sent, model))
    X_test_emb = X_test.apply(lambda sent: get_doc_vec(sent, model))
    
    # Handle missing/None embeddings with zero vectors
    zero_vector = np.zeros(model.vector_size)
    
    X_train_list = [emb if emb is not None else zero_vector for emb in X_train_emb]
    X_test_list = [emb if emb is not None else zero_vector for emb in X_test_emb]
    
    return np.array(X_train_list), np.array(X_test_list)


def extract_bert_embeddings(X_train, X_test, model_name="aubmindlab/bert-base-arabert"):
    """
    Extract CLS token embeddings using a pre-trained BERT model.
    """
    if torch is None:
        raise ImportError("Please install PyTorch and Transformers to use BERT.")
        
    print(f"Loading BERT model ({model_name})...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    def get_bert_embedding(text):
        if not isinstance(text, str) or len(text.strip()) == 0:
            text = " "
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.numpy()

    print("Generating BERT embeddings for Train set...")
    X_train_bert = np.vstack(X_train.apply(get_bert_embedding).values)
    
    print("Generating BERT embeddings for Test set...")
    X_test_bert = np.vstack(X_test.apply(get_bert_embedding).values)
    
    return X_train_bert, X_test_bert

