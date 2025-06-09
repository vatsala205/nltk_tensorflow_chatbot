import json
import random
import numpy as np
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress info and warning logs from TensorFlow


# Download nltk data if not already done
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents and model
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)


words, classes, X_train, y_train = None, None, None, None
with open('training_data.pkl', 'rb') as f:
    words, classes, X_train, y_train = pickle.load(f)

model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25  # You can tweak this threshold if you want
    results = [(i, r) for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that. Can you try rephrasing?"

    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])


def chatbot_response(text):
    ints = predict_class(text)
    res = get_response(ints, intents)
    return res


# For quick testing:
if __name__ == "__main__":
    print("Start chatting with the bot (type quit to stop)!")
    while True:
        message = input("You: ")
        if message.lower() == "quit":
            break
        response = chatbot_response(message)
        print("Bot:", response)
