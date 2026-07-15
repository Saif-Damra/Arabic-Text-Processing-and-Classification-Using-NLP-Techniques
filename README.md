# 📝 Arabic Text Processing and Classification Using NLP Techniques

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?logo=TensorFlow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-orange?logo=huggingface)

This project focuses on building an efficient Natural Language Processing (NLP) pipeline for Arabic text preprocessing and classification. It processes Arabic news headlines by removing noise (diacritics, punctuation, non-Arabic characters) and standardizing the text. Various machine learning and deep learning models were implemented and compared to classify the text effectively.

## 📑 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)

## 🔍 Overview

The main objective of this repository is to demonstrate an end-to-end NLP pipeline specifically tailored for the Arabic language. Arabic is morphologically rich and complex, requiring specialized preprocessing steps before applying standard classification algorithms. 

This project explores multiple text representation techniques (TF-IDF, Word2Vec, BERT) and compares their effectiveness across different classification algorithms.

## ✨ Key Features

- **Comprehensive Preprocessing:** Custom logic to remove Arabic diacritics (الحركات), English characters, stop words, and normalize specific Arabic letters (همزات، تاء مربوطة).
- **Multiple Embeddings:**
  - TF-IDF & CountVectorizer (One-Hot)
  - Pre-trained Word2Vec (Skip-Gram & CBoW)
  - AraBERT (`aubmindlab/bert-base-arabert`)
- **Diverse Models:**
  - Classical ML: Naive Bayes, Logistic Regression
  - Deep Learning: Bidirectional LSTM (Keras/TensorFlow)
  - Transformers: BERT Fine-Tuning (PyTorch)
- **Evaluation:** Comprehensive comparison using Accuracy and F1-macro average. BERT consistently shows the highest performance due to its bidirectional contextual understanding.

## 📁 Project Structure

```
Arabic-Text-Processing-and-Classification-Using-NLP-Techniques/
├── 📂 data/                    # Folder for dataset files (News_train.xlsx, etc.)
│   └── README.md               # Dataset instructions
├── 📂 models/                  # Folder for pre-trained Word2Vec models
│   └── README.md               # Models instructions
├── 📂 Notebooks/               # Original Jupyter notebooks for exploration
├── 📂 src/
│   ├── __init__.py
│   ├── main.py                 # Pipeline entry point
│   ├── data_processing.py      # Text cleaning and tokenization
│   ├── feature_extraction.py   # TF-IDF, W2V, and BERT embeddings
│   └── model_training.py       # ML, LSTM, and BERT training logic
├── .gitignore
├── README.md
└── requirements.txt
```

## 🧠 Methodology

1. **Text Preprocessing**: The text is cleaned using regex to strip unwanted characters and `nltk` to remove stop words.
2. **Feature Extraction**: The text is converted into numerical vectors. We heavily rely on `gensim` for Word2Vec and `transformers` for BERT embeddings.
3. **Classification**: 
   - TF-IDF / W2V -> Logistic Regression / Naive Bayes
   - Embedded sequences -> Bi-LSTM
   - Tokenized sequences -> Fine-tuned BERT sequence classifier

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
# Clone the repository
git clone https://github.com/Saif-Damra/Arabic-Text-Processing-and-Classification-Using-NLP-Techniques.git
cd Arabic-Text-Processing-and-Classification-Using-NLP-Techniques

# Install dependencies
pip install -r requirements.txt

# Download NLTK resources
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Data & Model Requirements
Due to file size limits, datasets and pre-trained Word2Vec models are not included. 
- Place `News_train.xlsx` and `News_test.xlsx` inside the `data/` folder.
- Place your Word2Vec models (`wikipedia_sg_300`, `wikipedia_cbow_300`) inside the `models/word2vec/` folder. 
*(See the respective READMEs in those folders for exact paths).*

## 💻 Usage

Run the main pipeline script from the terminal. The script is modular, allowing you to choose which feature extraction technique and model you wish to evaluate.

```bash
# Example: Train Logistic Regression using TF-IDF
python src/main.py --feature tfidf --model lr

# Example: Train LSTM using Word2Vec Skip-Gram
python src/main.py --feature w2v-sg --model lstm

# Example: Fine-tune AraBERT
python src/main.py --feature bert-tokens --model bert-ft
```

---
**Developed by [Saif Damra](https://github.com/Saif-Damra)**
