"""
Main Entry Point
=================
Arabic Text Processing and Classification Pipeline.

Usage:
    python src/main.py --feature tfidf --model lr
    python src/main.py --feature w2v-sg --model nb
    python src/main.py --feature bert-tokens --model bert-ft
"""

import argparse
import sys
import os
import warnings

warnings.filterwarnings("ignore")

# Ensure the root of the project is in the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing import load_and_preprocess_data
from src.feature_extraction import (
    extract_tfidf, extract_count_vectorizer, 
    extract_word2vec, extract_bert_embeddings
)
from src.model_training import train_ml_model, train_lstm_model, fine_tune_bert


def parse_args():
    parser = argparse.ArgumentParser(description="Arabic Text Classification Pipeline")
    parser.add_argument(
        "--train-data", type=str, default="data/News_train.xlsx",
        help="Path to training data Excel file"
    )
    parser.add_argument(
        "--test-data", type=str, default="data/News_test.xlsx",
        help="Path to testing data Excel file"
    )
    parser.add_argument(
        "--feature", type=str, choices=["tfidf", "onehot", "w2v-sg", "w2v-cbow", "bert-cls", "bert-tokens"],
        default="tfidf", help="Feature extraction method"
    )
    parser.add_argument(
        "--model", type=str, choices=["nb", "lr", "lstm", "bert-ft"],
        default="lr", help="Model to train (nb: Naive Bayes, lr: Logistic Regression, lstm: Bi-LSTM, bert-ft: BERT Fine-Tuning)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("🚀 ARABIC NLP CLASSIFICATION PIPELINE")
    print("=" * 60)
    
    # 1. Load and Preprocess Data
    try:
        X_train_text, X_test_text, y_train, y_test, label_encoder, train_df, test_df = load_and_preprocess_data(
            args.train_data, args.test_data
        )
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
        
    target_names = label_encoder.classes_
    print(f"\nClasses found: {list(target_names)}")
    
    # 2. Extract Features & 3. Train Model
    if args.model in ["nb", "lr"]:
        if args.feature == "tfidf":
            X_train_feat, X_test_feat, _ = extract_tfidf(X_train_text, X_test_text)
        elif args.feature == "onehot":
            X_train_feat, X_test_feat, _ = extract_count_vectorizer(X_train_text, X_test_text)
        elif args.feature == "w2v-sg":
            X_train_feat, X_test_feat = extract_word2vec(X_train_text, X_test_text, "models/word2vec/wiki_sg_300/wikipedia_sg_300")
        elif args.feature == "w2v-cbow":
            X_train_feat, X_test_feat = extract_word2vec(X_train_text, X_test_text, "models/word2vec/wiki_cbow_300/wikipedia_cbow_300")
        elif args.feature == "bert-cls":
            X_train_feat, X_test_feat = extract_bert_embeddings(X_train_text, X_test_text)
        else:
            print("\n❌ Error: Invalid feature choice for ML models. Cannot use 'bert-tokens'.")
            sys.exit(1)
            
        train_ml_model(args.model, X_train_feat, y_train, X_test_feat, y_test, target_names)
        
    elif args.model == "lstm":
        print("\nNote: LSTM requires embeddings to initialize its Embedding layer.")
        if args.feature == "tfidf":
            _, _, vectorizer = extract_tfidf(X_train_text, X_test_text)
            # This is simplified - TF-IDF vectors are not word embeddings, but the original code 
            # passed the TFIDF matrix weights to the Embedding layer. 
            # We'll extract a dummy matrix for compatibility, though W2V is highly recommended.
            embeddings = np.random.rand(4000, 300) 
            print("⚠️ Warning: TF-IDF is not typically used as weights for an Embedding layer. Using dummy weights. Use W2V for proper results.")
        elif args.feature == "w2v-sg":
            print("Loading Word2Vec model for LSTM initialization...")
            import gensim
            model_w2v = gensim.models.Word2Vec.load("models/word2vec/wiki_sg_300/wikipedia_sg_300")
            embeddings = model_w2v.wv.vectors
        else:
            print("\n❌ Error: Please use 'w2v-sg' as feature for LSTM in this simplified script.")
            sys.exit(1)
            
        train_lstm_model(X_train_text, y_train, X_test_text, y_test, embeddings, target_names)
        
    elif args.model == "bert-ft":
        if args.feature != "bert-tokens":
            print("\n⚠️ Warning: Feature argument ignored. Fine-tuning BERT tokenizes text directly.")
        fine_tune_bert(X_train_text, y_train, X_test_text, y_test, target_names)

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
