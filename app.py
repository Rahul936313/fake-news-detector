import streamlit as st
import pandas as pd
import joblib

# ===============================
# Load Model and Vectorizer
# ===============================
@st.cache_resource
def load_model():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return vectorizer, model

# ===============================
# Streamlit UI
# ===============================
def main():
    st.title("📰 Fake News Detector (ML Only)")

    vec, model = load_model()

    news_text = st.text_area("Paste a news article or headline:")

    if st.button("Check News"):
        if not news_text.strip():
            st.warning("⚠️ Please enter some news text.")
        else:
            # ML Prediction
            X_vec = vec.transform([news_text])
            pred = model.predict(X_vec)[0]
            prob = model.predict_proba(X_vec)[0]

            st.subheader(f"🤖 ML Prediction: **{pred}**")
            st.write(f"Probability: Fake={prob[0]:.2f}, Real={prob[1]:.2f}")

if __name__ == "__main__":
    main()
