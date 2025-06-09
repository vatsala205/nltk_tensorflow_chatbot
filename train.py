import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import pickle

# Load the training data (make sure you run preprocess.py first and save the data)
with open('training_data.pkl', 'rb') as f:
    words, classes, X_train, y_train = pickle.load(f)


# Build the model architecture
model = Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))  # input layer
model.add(Dropout(0.5))  # helps prevent overfitting
model.add(Dense(64, activation='relu'))  # hidden layer
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))  # output layer, number of classes

# Compile the model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model`
history = model.fit(np.array(X_train), np.array(y_train), epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5')
print("Model trained and saved as chatbot_model.h5")
