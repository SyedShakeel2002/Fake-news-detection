import streamlit as st
import numpy as np
import pickle
import re
import string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load LSTM model and tokenizer
model = load_model('lstm_fake_news_model.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Constants
maxlen = 500
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Streamlit UI
st.set_page_config(page_title="Fake News Detection (LSTM)", page_icon="🧠")
st.title("📰 Fake News Detection App (LSTM Model)")
st.markdown("Enter a news article below to check if it's **Reliable**, **Suspicious**, or **Unreliable (Fake)**.")

user_input = st.text_area("✍️ Paste news article here:", height=200)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.error("⚠️ Please enter some news content to classify.")
    else:
        # Step 1: Clean and preprocess
        cleaned = clean_text(user_input)
        sequence = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=maxlen)

        # Step 2: Predict
        prediction = model.predict(padded)
        score = float(prediction[0][0])
        label = "Fake" if score >= 0.5 else "Real"

        # Step 3: Show results
        st.markdown(f"### 🧠 Prediction Score: `{score:.4f}`")
        st.markdown(f"### 🏷️ Raw Classification: `{label}`")

        # Step 4: Three-level confidence-based label
        if score >= 0.9:
            st.warning("⚠️ This news is likely **Unreliable (Fake)**.")
        elif score >= 0.6:
            st.info("🤔 This news might be **Suspicious**, review with caution.")
        else:
            st.success("✅ This news is likely **Reliable (Real)**.")
