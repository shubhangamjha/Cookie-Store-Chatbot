import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import random
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

stemmer = LancasterStemmer()

with open('C:\\Users\shubh\\Desktop\\Desktop\\programming\\Chatbot\\json file\\intents.json') as file:
    data = json.load(file)

try:
    with open('C:\\Users\\shubh\\Desktop\\Desktop\\programming\\Chatbot\\data.pickle', 'rb') as f:
        words, labels, training, output = pickle.load(f)
except FileNotFoundError:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)

    training = np.array(training)
    output = np.array(output)

    with open('C:\\Users\\shubh\\Desktop\\Desktop\\programming\\Chatbot\\data.pickle', 'wb') as f:
        pickle.dump((words, labels, training, output), f)

# Build the neural network model
model = Sequential()
model.add(Dense(8, input_shape=(len(training[0]),), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(len(output[0]), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

try:
    model.load_weights('C:\\Users\\shubh\\Desktop\\Desktop\\programming\\Chatbot\\model_keras.h5')
except FileNotFoundError:
    # Train the model
    model.fit(training, output, epochs=50000, batch_size=100)

    # Save the model
    model.save_weights('C:\\Users\\shubh\\Desktop\\Desktop\\programming\\Chatbot\\model_keras.h5')


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]



    for se in s_words:
        for i, w in enumerate(words):
            if w==se:
                bag[i] = 1

    return np.array([bag])

def chat():
    print("Start Talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == 'quit':
            break

        results = model.predict(bag_of_words(inp, words))
        results_index = np.argmax(results)
        tag = labels[results_index]
        print("The tag is ",tag)

        for tg in data['intents']:
            if tg['tag'] == tag:
                response = tg['responses']

        print(random.choice(response))

chat()