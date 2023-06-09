import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load the text data
with open('path_to_text_data.txt', 'r', encoding='utf-8') as file:  # Add the path to your text data file
    text_data = file.read()

# Preprocess the text data
text_data = text_data.lower()

# Create a mapping of unique characters to integers
chars = sorted(list(set(text_data)))
char_to_int = {char: i for i, char in enumerate(chars)}
int_to_char = {i: char for i, char in enumerate(chars)}

# Create the training data
seq_length = 100
x_data = []
y_data = []
for i in range(0, len(text_data) - seq_length, 1):
    input_seq = text_data[i:i + seq_length]
    output_seq = text_data[i + seq_length]
    x_data.append([char_to_int[char] for char in input_seq])
    y_data.append(char_to_int[output_seq])

# Reshape the training data
n_patterns = len(x_data)
x_data = np.reshape(x_data, (n_patterns, seq_length, 1))
x_data = x_data / float(len(chars))
y_data = tf.keras.utils.to_categorical(y_data)

# Create the model architecture
model = Sequential()
model.add(LSTM(256, input_shape=(x_data.shape[1], x_data.shape[2]), return_sequences=True))
model.add(LSTM(256))
model.add(Dense(y_data.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(x_data, y_data, batch_size=128, epochs=50)

# Generate text
start_index = np.random.randint(0, len(x_data) - 1)
pattern = x_data[start_index]
generated_text = ''
for i in range(500):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(chars))
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    generated_text += result
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

# Print the generated text
print(generated_text)
