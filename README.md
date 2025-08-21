# 📰 Fake News Detector

A **Machine Learning + Streamlit** app to detect fake news using Logistic Regression and TF-IDF vectorization.  
This project was built as part of my learning journey in AI & Data Science.

---

## 🚀 Features
- Logistic Regression model trained on **Fake vs Real news dataset**
- Text preprocessing using **TF-IDF**
- Simple **Streamlit interface**
- Prediction with probability scores (Fake / Real)

---

## 📂 Project Structure
news_detection/
│── app.py # Streamlit web app
│── train_model.py # Script to train model and save .pkl files
│── merged.py # Script to clean & merge datasets
│── datasets/ # Raw datasets (ignored in .gitignore)
│── fake_news_model.pkl # Saved trained model (ignored in .gitignore)
│── vectorizer.pkl # Saved TF-IDF vectorizer (ignored in .gitignore)
│── merged_dataset.csv # Clean merged dataset (ignored in .gitignore)
│── requirements.txt # Dependencies
│── README.md # Project documentation


---

## ⚡ How to Run Locally
1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/fake-news-detector.git
   cd fake-news-detector

2. Install dependencies:
=> pip install -r requirements.txt

3.Train the model (optional, pre-trained model is already saved):
 => python train_model.py

4.Run the Streamlit app:
=> streamlit run app.py














