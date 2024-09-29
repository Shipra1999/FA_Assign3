#!/usr/bin/env python
# coding: utf-8

# In[1]:



# In[2]:


import streamlit as st
import joblib
import re
import requests
from bs4 import BeautifulSoup

# Load the pre-trained models (make sure these files exist in the same directory)
try:
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    rf_model = joblib.load('random_forest_model.pkl')
except FileNotFoundError:
    st.error("Model files 'tfidf_vectorizer.pkl' and/or 'random_forest_model.pkl' not found. Please check the file paths.")

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Function to extract text from URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Extract all text from paragraph tags
            paragraphs = soup.find_all('p')
            extracted_text = ' '.join([para.get_text() for para in paragraphs])
            return extracted_text
        else:
            st.error(f"Error retrieving content from URL: Status code {response.status_code}")
            return None
    except requests.RequestException as e:
        st.error(f"Error fetching URL: {e}")
        return None

st.title("Text Origin Classification (MGT vs HGT)")

# User input for text
user_input = st.text_area("Enter text to classify:")
if st.button("Classify Text"):
    if user_input:
        preprocessed_input = preprocess_text(user_input)
        input_tfidf = tfidf.transform([preprocessed_input])
        prediction = rf_model.predict(input_tfidf)
        st.write(f"The text is: {'Machine Generated' if prediction == 1 else 'Human Generated'}")
    else:
        st.warning("Please enter some text to classify.")

# User input for URL
url_input = st.text_input("Enter a URL to classify:")
if st.button("Classify URL"):
    if url_input:
        text_from_url = extract_text_from_url(url_input)
        if text_from_url:
            preprocessed_url_text = preprocess_text(text_from_url)
            url_tfidf = tfidf.transform([preprocessed_url_text])
            prediction = rf_model.predict(url_tfidf)
            st.write(f"The content from the URL is: {'Machine Generated' if prediction == 1 else 'Human Generated'}")
        else:
            st.write("Unable to extract text from the URL.")
    else:
        st.warning("Please enter a valid URL.")





