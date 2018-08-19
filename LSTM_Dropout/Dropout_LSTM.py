from keras.datasets import imdb

vocabulary_size = 7000
"""
Setting vocabulary_size to 7000 trims the dataset length to get
only 7000 words from the dataset to load into our variables.
"""

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocabulary_size)

from keras.preprocessing import sequence

max_words = 400
X_train = sequence.pad_sequences(X_train, maxlen=max_words) # padding the max_number of words we take in for reviews for training
X_test = sequence.pad_sequences(X_test, maxlen=max_words) # padding the max_number of words we take in for reviews for testing.

"""
Padding makes sure that reviews lenghts are same in the input vector
by truncating the longer reviews and adding 0's where the review length
is shorter than 400 words.

"""

from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, MaxPool1D, Conv1D, Conv2D

import numpy as np
seed = 8 # Seed is set to 8 to provide consistency in training 
np.random.seed(8)

model = Sequential()
embedding_size=64
model=Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(Dropout(0.25))
model.add(LSTM(200))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=128, verbose=1)

evaluation = model.evaluate(X_test, y_test)
print("Accuracy %f%%" % (evaluation[1] * 100))
