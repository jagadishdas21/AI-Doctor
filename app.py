from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
from flask_cors import CORS
import os
import logging

# Initialize the Flask app and configure CORS
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for CORS

# Enable logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Load your chatbot model and required files (absolute paths)
model_path = os.path.join(os.getcwd(), 'chatbot_model.h5')
model = load_model(model_path)

# Load words and classes
with open(os.path.join(os.getcwd(), 'words.pkl'), 'rb') as f:
    words = pickle.load(f)
with open(os.path.join(os.getcwd(), 'classes.pkl'), 'rb') as f:
    classes = pickle.load(f)

# Load intents using json.load()
with open(os.path.join(os.getcwd(), 'intents.json'), 'r') as f:
    intents = json.load(f)

lemmatizer = WordNetLemmatizer()

# Function to clean up and process user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Create a bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

# Predict the class of the user input
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    
    # Logging the model's raw predictions
    logging.debug(f"Raw predictions: {res}")
    
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    
    # Logging results after sorting
    logging.debug(f"Sorted results: {results}")
    
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


# Get response based on predicted class
def get_response(intents_list):
    tag = intents_list[0]['intent']
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])
    return "Sorry, I didn't understand that."

# Main chatbot function to generate response
def chatbot_response(message):
    intents_list = predict_class(message)
    if intents_list:
        response = get_response(intents_list)
    else:
        response = "I'm sorry, I didn't understand that. Could you rephrase?"
    return response

# Serve the homepage
@app.route('/')
def index():
    return render_template('index.html')  # Frontend HTML for chatbot UI

# Handle user message input and return chatbot response
@app.route('/get', methods=['POST'])
def chat():
    try:
        logging.info("Chat request received")

        # Get the user input message from the request
        user_input = request.form.get('msg')
        if not user_input:
            logging.error("No message in request")
            return jsonify({'error': 'No message found'}), 400

        logging.info(f"Received message: {user_input}")
        
        # Generate chatbot response
        response = chatbot_response(user_input)
        logging.info(f"Chatbot response: {response}")
        
        return jsonify({'response': response}), 200
    
    except Exception as e:
        logging.error(f"Error during chatbot interaction: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Run the application on the specified host and port
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)