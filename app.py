import streamlit as st
import tensorflow as tf
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load model
model = tf.keras.models.load_model("spam_lstm_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# UI
st.title("📩 SMS Spam Detection")
st.write("Live Demo – Spam-Klassifikation mit LSTM")

user_input = st.text_area("SMS Text eingeben")

if st.button("Vorhersage"):
    clean_text = preprocess_text(user_input)
    seq = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(seq, maxlen=100)
    prediction = model.predict(padded)[0][0]

    if prediction > 0.5:
        st.error(f"🚨 Spam (Wahrscheinlichkeit: {prediction:.2f})")
    else:
        st.success(f"✅ Ham (Wahrscheinlichkeit: {1 - prediction:.2f})")
