"""
Model Training Module
======================
Handles training and evaluation of models:
1. Classical ML (Naive Bayes, Logistic Regression)
2. Bi-LSTM (Keras)
3. BERT Fine-Tuning (PyTorch)
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    tf = None

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import BertTokenizer, AutoModelForSequenceClassification, AdamW
except ImportError:
    torch = None


def train_ml_model(model_type: str, X_train, y_train, X_test, y_test, target_names=None):
    """
    Train and evaluate a classical ML model.
    """
    print(f"\n--- Training {model_type.upper()} ---")
    if model_type == 'nb':
        model = GaussianNB()
    elif model_type == 'lr':
        model = LogisticRegression(max_iter=1000)
    else:
        raise ValueError("Invalid model_type. Use 'nb' or 'lr'.")
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    return model


def train_lstm_model(X_train_text, y_train, X_test_text, y_test, embeddings, target_names=None):
    """
    Train and evaluate a Bi-LSTM model using Keras.
    Requires text for tokenization and pre-extracted embeddings to initialize the Embedding layer.
    """
    if tf is None:
        raise ImportError("Please install TensorFlow to use LSTM.")
        
    print("\n--- Training Bi-LSTM ---")
    
    # Tokenize text
    vocab_size = 4000
    max_length = 100
    
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(X_train_text)
    
    seq_train = tokenizer.texts_to_sequences(X_train_text)
    seq_test = tokenizer.texts_to_sequences(X_test_text)
    
    pad_train = pad_sequences(seq_train, maxlen=max_length, padding='pre')
    pad_test = pad_sequences(seq_test, maxlen=max_length, padding='pre')
    
    # Convert labels to categorical
    num_classes = len(np.unique(y_train))
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
    
    # Ensure embeddings fit vocab size
    emb_dim = embeddings.shape[1]
    # Simple fix if embeddings matrix is smaller than vocab size
    if embeddings.shape[0] < vocab_size:
        print(f"Warning: Embedding matrix shape {embeddings.shape} is smaller than vocab size {vocab_size}.")
        pad = np.zeros((vocab_size - embeddings.shape[0], emb_dim))
        embeddings_matrix = np.vstack([embeddings, pad])
    else:
        embeddings_matrix = embeddings[:vocab_size, :]
        
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=max_length, 
                  weights=[embeddings_matrix], trainable=False),
        Bidirectional(LSTM(100, return_sequences=False)),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    
    print("Fitting model...")
    model.fit(pad_train, y_train_cat, epochs=10, batch_size=32, verbose=1, validation_data=(pad_test, y_test_cat))
    
    print("\nEvaluating model...")
    val_loss, val_accuracy = model.evaluate(pad_test, y_test_cat, verbose=0)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    y_pred = model.predict(pad_test)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred_labels, target_names=target_names))
    
    return model


def fine_tune_bert(X_train_text, y_train, X_test_text, y_test, target_names=None, model_name="aubmindlab/bert-base-arabert"):
    """
    Fine-tune a BERT model for sequence classification.
    """
    if torch is None:
        raise ImportError("Please install PyTorch and Transformers to fine-tune BERT.")
        
    print(f"\n--- Fine-tuning {model_name} ---")
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    num_labels = len(np.unique(y_train))
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # Find max sequence length safely
    all_text = X_train_text.tolist() + X_test_text.tolist()
    max_sequence_len = min(512, max([len(str(x).split()) for x in all_text]))
    print(f"Using max sequence length: {max_sequence_len}")
    
    def tokenize_data(sentences):
        # Convert non-string to empty space
        clean_sents = [str(s) if isinstance(s, str) else " " for s in sentences]
        return tokenizer(clean_sents, padding=True, truncation=True, return_tensors="pt", max_length=max_sequence_len)
        
    print("Tokenizing train data...")
    train_tokens = tokenize_data(X_train_text.tolist())
    print("Tokenizing test data...")
    test_tokens = tokenize_data(X_test_text.tolist())
    
    train_dataset = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'], torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(test_tokens['input_ids'], test_tokens['attention_mask'], torch.tensor(y_test, dtype=torch.long))
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = [item.to(device) for item in batch]
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}")
        
    print("\nEvaluating model...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [item.to(device) for item in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            
            predictions = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names))
    
    return model
