import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from JSON
intents = json.loads(open('D:\\Personal Projects\\Chatbot Using Python\\intents.json').read())

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

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

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

# Split the data into training and validation sets
trainX, valX, trainY, valY = train_test_split(training[:, :len(words)], training[:, len(words):], test_size=0.2, random_state=42)

# Build model with improved architecture
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(256, input_shape=(len(trainX[0]),), activation='relu'))  # Increased neurons
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu'))  # Additional layer for complexity
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# Compile model with Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Implement Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit the model with validation data
hist = model.fit(np.array(trainX), np.array(trainY), 
                  epochs=300, batch_size=5, verbose=1, 
                  validation_data=(np.array(valX), np.array(valY)), 
                  callbacks=[early_stopping])  # Include early stopping

# Save the model
model.save('chatbot_model.keras')
print('Model trained and saved!')
