import streamlit as st
import pickle
from utils import clean_text  # ✅ Import the cleaning function

# Load model and vectorizer
model = pickle.load(open("models/spam_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# Streamlit page settings
st.set_page_config(page_title="Email Spam Detector")
st.title("📧 Email Spam Detector")

# Input box
input_text = st.text_area("Enter your email/message here:")

# Prediction
if st.button("Check"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        # ✅ Preprocess input before prediction
        cleaned = clean_text(input_text)
        transformed = vectorizer.transform([cleaned])
        result = model.predict(transformed)[0]

        # Display result
        if result == 0:
            st.error("❌ SPAM detected!")
        else:
            st.success("✅ HAM (Not Spam)")

