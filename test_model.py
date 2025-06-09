import numpy as np
import pickle
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import nltk

# Download nltk data if not yet
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load data
with open('training_data.pkl', 'rb') as f:
    words, classes, X_train, y_train = pickle.load(f)

# Load model
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

# Test input
input_sentence = "account not working"
bow = bag_of_words(input_sentence, words)
res = model.predict(np.array([bow]))[0]
predicted_class_index = np.argmax(res)
predicted_tag = classes[predicted_class_index]

print(f"Input: {input_sentence}")
print(f"Predicted tag: {predicted_tag}")
