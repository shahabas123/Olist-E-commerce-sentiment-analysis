# Olist-E-commerce-sentiment-analysis

# 🛒 Olist E-Commerce Sentiment Analysis

This project performs **sentiment analysis** on customer reviews from the [Brazilian E-Commerce (Olist) dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), using Natural Language Processing (NLP) and machine learning models. The goal is to classify reviews as **Positive**, **Neutral**, or **Negative** based on the review text.

---

## 📁 Dataset

- Source: Kaggle - Olist Brazilian E-Commerce Public Dataset
- Used file: `olist_order_reviews_dataset.csv` (25,000 reviews sampled)

---

## 🧠 Problem Statement

Customer reviews in the dataset are unstructured and in **Portuguese**. This project builds a sentiment classification system using translated reviews to automate and analyze customer feedback.

---

## 🔍 Key Features

- ✅ Translates Portuguese reviews to English using `googletrans`
- ✅ Handles missing review texts with score-based imputation
- ✅ Text preprocessing: punctuation removal, stopword removal, stemming
- ✅ Feature extraction using **TF-IDF Vectorization**
- ✅ Sentiment labeling based on `review_score`:
  - `5, 4` → Positive
  - `3` → Neutral
  - `1, 2` → Negative
- ✅ Class balancing using **SMOTE**
- ✅ Model training with:
  - Random Forest
  - AdaBoost
  - Multinomial Naive Bayes (optional)
- ✅ Model and vectorizer saved using `pickle`
- ✅ Predicts sentiment for custom inputs like `"Excellent product!"`

---

## 🛠️ Tech Stack

| Component         | Tool/Library            |
|------------------|-------------------------|
| Language          | Python                  |
| Notebook          | Google Colab            |
| NLP               | NLTK, Scikit-learn, googletrans |
| Modeling          | Random Forest, AdaBoost |
| Resampling        | imbalanced-learn (SMOTE)|
| Deployment-ready  | Model saved as `.pkl`   |
| Version Control   | Git, GitHub, Git LFS    |

---

## 📊 Model Performance

- Best model: **Random Forest**
- Accuracy (with SMOTE): ~78%
- Evaluation Metrics: Precision, Recall, F1-Score

---

## 🚀 How to Run

1. Clone the repository:


git clone https://github.com/shahabas123/Olist-E-commerce-sentiment-analysis.git
cd Olist-E-commerce-sentiment-analysis


2.Set up dependencies:
pip install -r requirements.txt


3. Download the pre-trained model and vectorizer from the **[Releases](https://github.com/shahabas123/Olist-E-commerce-sentiment-analysis/releases)** section of this repository.

Place them in your project directory:

- `rf_model.pkl`
- `tfidf_vectorizer.pkl`

4. Run prediction using pickle
 import pickle

# Load model and vectorizer
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Predict sentiment
text = ["excellent product"]
vec_text = vectorizer.transform(text)
pred = model.predict(vec_text)

print(pred)  # Output: Positive / Neutral / Negative


Sample prediction:
Input: "Excellent product!"
Output: Positive

overview

📈 Streamlit Web App
This project also includes a Streamlit-based web application with a user-friendly interface. It has the following pages:

Home Page: Introduction and basic description of the sentiment classification.

Prediction Page: Enter a custom review and get the sentiment prediction instantly.

Analysis Page: Offers in-depth visual insights including:

📉 Sentiment trend over time

🚨 Negative sentiment alerts

📦 Worst-performing product categories

This makes the project suitable for both end-users and business stakeholders looking to monitor and improve customer satisfaction.

 Acknowledgements
 
Olist on Kaggle

Scikit-learn

Google Translate API via googletrans

Streamlit

📬 Contact

Shahabas Ali
Email: shahabas751@gmail.com
GitHub: @shahabas123

