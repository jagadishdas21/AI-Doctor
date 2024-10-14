import random
import json
import numpy as np
import nltk
import pickle
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.optimizers import Adam

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from JSON
def load_intents(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Ensure NLTK resources are downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Path to intents.json
intents_file_path = 'intents.json'

# Load intents
try:
    intents = load_intents(intents_file_path)
except Exception as e:
    print(f"Error loading intents: {e}")
    raise

# Initialize lists for training data
words = []
classes = []
documents = []
ignoreLetters = ['?', '!', '.', ',']

# Tokenization and data preparation
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatization and filtering
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes to files
with open('words.pkl', 'wb') as f:
    pickle.dump(words, f)
with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

# Prepare training data
training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]

    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

# Split into train and test data
trainX = np.array([i[:len(words)] for i in training])
trainY = np.array([i[len(words):] for i in training])

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, input_shape=(len(trainX[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(trainY[0]), activation='softmax')
])

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Set up early stopping to monitor validation loss
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    trainX, 
    trainY, 
    epochs=500,  # Increased epochs
    batch_size=5, 
    verbose=1, 
    validation_split=0.2, 
    callbacks=[early_stopping]
)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(
    np.array(trainX), 
    np.array(trainY), 
    epochs=300, 
    batch_size=5, 
    verbose=1, 
    validation_split=0.2
)

# Save the model
model.save('chatbot_model.h5')

# Load model and necessary files for chatbot use
def load_model_and_data():
    model = tf.keras.models.load_model('chatbot_model.h5')
    with open('words.pkl', 'rb') as f:
        words = pickle.load(f)
    with open('classes.pkl', 'rb') as f:
        classes = pickle.load(f)
    with open('intents.json', 'r') as f:
        intents = json.load(f)
    return model, words, classes, intents

# Functions for processing input and predicting response
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])

# Chatbot response generation
def chatbot_response(message):
    intents_list = predict_class(message, model)
    if intents_list:
        response = get_response(intents_list, intents)
    else:
        response = "I'm sorry, I didn't understand that. Could you rephrase?"
    return response