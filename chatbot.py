import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load necessary files
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')  # Make sure to use the correct model filename

# Function to clean up user input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Create a bag of words from user input
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
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# Get a response from the chatbot based on the predicted class
def get_response(intents_list):
    tag = intents_list[0]['intent']
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that."

# Main function to interact with the user
def chatbot_response(message):
    intents_list = predict_class(message)
    if intents_list:
        response = get_response(intents_list)
    else:
        response = "I’m sorry, I didn’t understand that. Could you please rephrase?"
    return response

# Start the chatbot in terminal
if __name__ == "__main__":
    print("Chatbot is running! Type 'exit' to stop.")
    
    while True:
        user_input = input("You: ")  # Take user input
        if user_input.lower() == 'exit':  # Exit condition
            print("Chatbot: Goodbye!")
            break
        
        response = chatbot_response(user_input)  # Get the response from the chatbot
        print(f"Chatbot: {response}")  # Print the chatbot's response
