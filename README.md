# Arabic Text Processing and Classification Using NLP Techniques

This project focuses on building an efficient natural language processing (NLP) pipeline for Arabic text preprocessing and classification. The preprocessing steps involve removing diacritics, punctuation, and non-Arabic characters, along with tokenizing text and standardizing certain Arabic letters. Various machine learning models were implemented to classify news headlines in Arabic, with an emphasis on evaluating performance using accuracy and F1 scores.

## Key Features:
- Comprehensive Arabic text preprocessing: removal of diacritics, stop words, and punctuation, tokenization, and conversion of specific Arabic letters.
- Multiple models (BERT, Word2Vec Skip Gram, Logistic Regression, LSTM) are used for text classification.
- Comparison of performance between models using accuracy and F1-macro average.
- BERT model shows the highest performance due to its bidirectional contextual understanding.
- Evaluation of word representation techniques and model effectiveness.

## Dataset:
The dataset primarily consists of Arabic news headlines. Preprocessing steps were applied to clean and standardize the data, making it suitable for classification tasks. Model performance is evaluated on this dataset, with BERT outperforming other models in both accuracy and F1 score.
