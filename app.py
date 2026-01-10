import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = tf.keras.models.load_model("spam_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 40

st.title("📩 Spam SMS Detector")
st.write("Enter an SMS message:")

text = st.text_area("SMS Text")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter a message.")
    else:
        seq = tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=max_len, padding="post")

        prob = model.predict(pad)[0][0]

        st.write("Spam probability:", round(prob, 3))

        if prob > 0.5:
            st.error("🚨 SPAM")
        else:
            st.success("✅ HAM")

