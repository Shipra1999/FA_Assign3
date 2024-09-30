from flask import Flask, request, jsonify, render_template
import pickle

# Load the Random Forest model and TF-IDF Vectorizer
with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Preprocess text function
def preprocess_text(text):
    return text  # Add your preprocessing logic here

# Function to extract text from a URL (dummy implementation)
def extract_text_from_url(url):
    return "Sample text from the URL"  # Replace with actual extraction logic

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML form

@app.route('/classify_text', methods=['POST'])
def classify_text():
    user_input = request.form['user_input']
    preprocessed_input = preprocess_text(user_input)
    input_tfidf = tfidf.transform([preprocessed_input])
    prediction = rf_model.predict(input_tfidf)
    result = 'Machine Generated' if prediction == 1 else 'Human Generated'
    return jsonify({'result': result})

@app.route('/classify_url', methods=['POST'])
def classify_url():
    url_input = request.form['url_input']
    text_from_url = extract_text_from_url(url_input)
    preprocessed_url_text = preprocess_text(text_from_url)
    url_tfidf = tfidf.transform([preprocessed_url_text])
    prediction = rf_model.predict(url_tfidf)
    result = 'Machine Generated' if prediction == 1 else 'Human Generated'
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
