# Rating Prediction with Sentiment Analysis

**Rating Prediction with Sentiment Analysis** is a deep learning–based system designed to predict **star-level ratings** from free-text customer reviews by integrating **sentiment analysis** with **advanced NLP techniques**.  
The project focuses on enhancing prediction accuracy while ensuring **data integrity** by removing spam/fake reviews and applying robust preprocessing steps.

---

## 🧭 Overview
Customer reviews often lack direct rating scores, making it difficult to extract structured feedback.  
This project addresses that challenge by:  
- 🧹 **Cleaning and preprocessing** large-scale e-commerce reviews  
- 🛠 **Handling missing values** using mode-based and **K-Nearest Neighbor (KNN) imputation**  
- 💬 **Applying sentiment analysis** to enhance star rating prediction  
- 🧠 **Training deep learning models** (Bi-LSTM, CNN, BERT) for review-based rating prediction  
- 🛡 **Filtering out fake/spam reviews** before training

---

## 📦 Dataset
- **Source:** Kaggle — Shopper Sentiments for TeePublic  
- **Size:** 278,100 free-text reviews with corresponding ratings (1–5 stars)  
- **Time Range:** 2018–2023  
- **Features:** Review text, rating, date, month, store location, longitude, latitude  
- **Preprocessing:**  
  - Removal of fake/spam reviews  
  - Text cleaning (HTML tags, punctuation, special characters)  
  - Tokenization & stopword removal  
  - Stemming (PorterStemmer)  
  - Vectorization (Word2Vec)  
  - Numerical scaling (StandardScaler)

---

## 🛠 Preliminary Requirements
- 🐍 Python **3.10+**
- 📦 Install required packages:
```bash
pip install numpy pandas scikit-learn nltk gensim tensorflow keras matplotlib seaborn
```

## 📊 Models & Evaluation Metrics:-

### 🧠 Models:
- **Bi-LSTM**
- **CNN**
- **BERT**

### 📏 Metrics:
- **Accuracy**
- **Precision, Recall, F1-score**
- **Confusion Matrix**
