<h1>**Twitter Sentiment Analysis — LSTM Deep Learning Model**</h1>

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-LSTM-red?logo=keras&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-96%25-brightgreen)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> **Classify the sentiment of tweets — Positive, Negative, or Neutral — using a Long Short-Term Memory (LSTM) deep learning model trained on real Twitter data.**

---

📌Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Pipeline](#-project-pipeline)
- [Model Architecture](#-model-architecture)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Key Learnings](#-key-learnings)
- [Author](#-author)

---

## 🧠 Overview

This project implements an end-to-end **Natural Language Processing (NLP)** pipeline to detect the sentiment expressed in tweets. Using deep learning — specifically a **Bidirectional LSTM** network — the model learns contextual patterns in tweet text and classifies them into sentiment categories.

The model achieved **96% classification accuracy** on the test set, demonstrating strong generalisation across varied tweet styles, slang, and abbreviations.

---

## 🎯 Problem Statement

Social media platforms like Twitter generate millions of text data points every day. Automatically understanding public sentiment from this data has real-world applications in:

- **Brand monitoring** — How do users feel about a product or company?
- **Market research** — Tracking public opinion on financial assets or events
- **Crisis detection** — Identifying spikes in negative sentiment around events
- **Political analysis** — Gauging public response to policies or candidates

Manual labelling is impractical at scale. This project builds an automated, high-accuracy sentiment classifier using deep learning.

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | Twitter public dataset (CSV format) |
| Target Labels | Positive / Negative / Neutral |
| Features | Raw tweet text |
| Preprocessing | Lowercasing, stopword removal, tokenisation, padding |

> The dataset consists of labelled tweets collected via the Twitter API / Kaggle's Sentiment140 or similar benchmark corpus. Each record contains a tweet and its corresponding sentiment label.

---

## 🔄 Project Pipeline

```
Raw Tweets
    │
    ▼
┌─────────────────────────┐
│   Data Collection &     │
│   Loading (CSV/API)     │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Text Preprocessing    │
│  • Lowercasing          │
│  • Remove URLs, @tags,  │
│    hashtags, punctuation│
│  • Stopword Removal     │
│  • Tokenisation         │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Sentiment Labelling   │
│  Positive / Negative /  │
│  Neutral                │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Sequence Encoding     │
│  • Word Embedding       │
│  • Padding / Truncation │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   LSTM Model Training   │
│  • Embedding Layer      │
│  • Bidirectional LSTM   │
│  • Dense + Softmax      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Evaluation            │
│  Accuracy · Precision   │
│  Recall · F1 · Confusion│
│  Matrix                 │
└─────────────────────────┘
```

---

## 🏗️ Model Architecture

```
Input Layer (Tokenised Tweet Sequences)
        │
        ▼
Embedding Layer (vocab_size × embedding_dim)
        │
        ▼
Bidirectional LSTM (units=128, return_sequences=True)
        │
        ▼
Dropout (rate=0.3)
        │
        ▼
LSTM (units=64)
        │
        ▼
Dropout (rate=0.3)
        │
        ▼
Dense (units=64, activation='relu')
        │
        ▼
Dense (units=num_classes, activation='softmax')
        │
        ▼
Output: Sentiment Class (Positive / Negative / Neutral)
```

**Why LSTM?**

LSTMs are designed to capture **long-range dependencies** in sequential data. Unlike traditional ML models (Naive Bayes, SVM), LSTMs understand word *order* and *context* — critical for understanding that "not good" is negative, not positive.

---

## 📈 Results

| Metric | Score |
|---|---|
| **Test Accuracy** | **96%** |
| Precision | High across all classes |
| Recall | Balanced across Positive / Negative / Neutral |
| Loss Function | Categorical Cross-Entropy |
| Optimiser | Adam |

The model significantly outperforms traditional bag-of-words approaches and demonstrates robust performance even on noisy, short-form social media text.

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow 2.x, Keras |
| NLP | NLTK, re (regex), Tokenizer |
| Data Handling | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/VinayakPA-3111/Twitter-Sentimental-Analysis.git
cd Twitter-Sentimental-Analysis
```

### 2. Install dependencies

```bash
pip install tensorflow keras pandas numpy nltk matplotlib seaborn scikit-learn jupyter
```

### 3. Download NLTK resources

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### 4. Launch the notebook

```bash
jupyter notebook twitter-sentiment-analysis-lstm.ipynb
```

### 5. Run all cells

Execute all cells in order — the notebook handles data loading, preprocessing, model training, and evaluation end-to-end.

---

## 📁 Project Structure

```
Twitter-Sentimental-Analysis/
│
├── twitter-sentiment-analysis-lstm.ipynb   # Main notebook (full pipeline)
├── README.md                               # Project documentation
│
└── data/                                   # (Add your dataset here)
    └── tweets.csv
```

> **Note:** The dataset file is not included in this repository. Download a sentiment-labelled Twitter dataset (e.g., Sentiment140 from Kaggle) and place it in the `data/` directory before running the notebook.

---

## 💡 Key Learnings

- **Text preprocessing is critical** — Removing noise (URLs, mentions, special characters) had a large positive impact on model accuracy.
- **Embeddings > Bag-of-Words** — Word embeddings preserve semantic relationships between words that one-hot encoding misses.
- **Bidirectional LSTMs** capture both past and future context within a sequence, improving classification on ambiguous tweets.
- **Dropout regularisation** was essential to prevent overfitting on the training data.
- Handling **class imbalance** (more neutral tweets than strongly positive/negative) required careful evaluation beyond raw accuracy.

---

## 🔭 Future Improvements

- [ ] Replace LSTM with a **BERT / RoBERTa** transformer model for higher contextual understanding
- [ ] Deploy as a **REST API** (FastAPI / Flask) for real-time tweet sentiment scoring
- [ ] Add a **Streamlit web app** for interactive sentiment prediction
- [ ] Integrate **Twitter API v2** for live tweet streaming and real-time classification
- [ ] Extend to **aspect-based sentiment analysis** (e.g., sentiment about specific product features)

---

## 👤 Author

**Vinayak Algundgi**

- 📧 [vinayak.algundgi12@gmail.com](mailto:vinayak.algundgi12@gmail.com)
- 💼 [LinkedIn](https://www.linkedin.com/in/vinayakalgundgi/)
- 🐙 [GitHub](https://github.com/VinayakPA-3111)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute with attribution.

---

*Built with curiosity, Python, and a lot of tweets. ⚡*
