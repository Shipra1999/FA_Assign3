#!/usr/bin/env python
# coding: utf-8

# In[1]:



# In[2]:


import streamlit as st
import re
import joblib
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load pre-trained TF-IDF vectorizer and Random Forest model
tfidf = joblib.load('tfidf_vectorizer.pkl')
rf_model = joblib.load('random_forest_model.pkl')

# Preprocess text function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()  # Remove leading/trailing whitespace
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Extract text from URL function
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([para.get_text() for para in paragraphs])
        return text
    except Exception as e:
        return None

# Streamlit App Title
st.title("Text Origin Classification (MGT vs HGT)")

# User input for text
user_input = st.text_area("Enter text to classify:")
if st.button("Classify Text"):
    preprocessed_input = preprocess_text(user_input)
    input_tfidf = tfidf.transform([preprocessed_input])
    prediction = rf_model.predict(input_tfidf)
    st.write(f"The text is: {'Machine Generated' if prediction == 1 else 'Human Generated'}")

# User input for URL
url_input = st.text_input("Enter a URL to classify:")
if st.button("Classify URL"):
    text_from_url = extract_text_from_url(url_input)
    if text_from_url:
        preprocessed_url_text = preprocess_text(text_from_url)
        url_tfidf = tfidf.transform([preprocessed_url_text])
        prediction = rf_model.predict(url_tfidf)
        st.write(f"The content from the URL is: {'Machine Generated' if prediction == 1 else 'Human Generated'}")
    else:
        st.write("Unable to extract text from the URL.")

# In[ ]:




