#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip3 install streamlit')


# In[2]:


import streamlit as st

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




