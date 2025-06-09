import json
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle

# Download necessary NLTK data files (run this once)
nltk.download('punkt')     # Tokenizer models
nltk.download('wordnet')   # Lemmatizer data

# Initialize the WordNet lemmatizer to reduce words to their base form
lemmatizer = WordNetLemmatizer()

# Load the intents JSON file which contains patterns and tags
with open('intents.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Create lists to hold words, classes (tags), and documents (pattern + tag pairs)
words = []
classes = []
documents = []

# List of characters to ignore (punctuation marks)
ignore_letters = ['?', '!', '.', ',']

# Loop through each intent in the JSON data
for intent in data['intents']:
    # Loop through each pattern in the current intent
    for pattern in intent['patterns']:
        # Tokenize each pattern into words
        word_list = nltk.word_tokenize(pattern)
        # Add tokenized words to the words list
        words.extend(word_list)
        # Add the tokenized pattern and its tag as a tuple to documents
        documents.append((word_list, intent['tag']))
        # Add the intent tag to classes list if not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize each word to its base form, convert to lowercase, and ignore punctuation
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]

# Remove duplicates and sort the word and class lists
words = sorted(set(words))
classes = sorted(set(classes))

# Print the processed word list, classes, and documents for verification
print(f"Words: {words}")
print(f"Classes: {classes}")
print(f"Documents: {documents}")

# Create training data (bag of words + output)
training = []

# Create an empty output array for each class (tag)
output_empty = [0] * len(classes)

# Loop through each document (pattern + tag)
for doc in documents:
    bag = []
    # Lemmatize and lowercase each word in the pattern
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in doc[0]]

    # Create a bag of words array: 1 if word exists in pattern, else 0
    for w in words:
        bag.append(1 if w in pattern_words else 0)

    # Create output row with 1 for the current tag, 0 for others
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    # Append bag of words and output row to the training data
    training.append([bag, output_row])

# Shuffle the training data to mix patterns and tags randomly
random.shuffle(training)

# Convert the training data to a numpy array for use in the model
training = np.array(training, dtype=object)

# Split training data into inputs (X) and outputs (y)
X_train = list(training[:, 0])
y_train = list(training[:, 1])

# Save the processed data
with open('training_data.pkl', 'wb') as f:
    pickle.dump((words, classes, X_train, y_train), f)

