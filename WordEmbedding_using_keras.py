from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, Flatten
from tensorflow.keras.models import Sequential
import numpy as np

# Example corpus
corpus = [
    "king queen man woman",
    "cat dog animal pet",
    "car bus train vehicle",
    "apple banana fruit food",
    "computer keyboard mouse technology"
]

# Step 1: Tokenize
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)
word_index = tokenizer.word_index
padded_sequences = pad_sequences(sequences, maxlen=10)

# Step 2: Build model
model = Sequential([
    Embedding(input_dim=1000, output_dim=64, input_length=10),
    Flatten(),
    Dense(8, activation='sigmoid'),
    Dense(1, activation='sigmoid')
])

# Step 3: Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Dummy labels for training
labels = np.random.randint(0, 2, size=(len(padded_sequences), 1))
history = model.fit(padded_sequences, labels, epochs=30, verbose=0)
print(model.summary())
print(history.history['accuracy'])

# Step 5: Get embeddings
embeddings = model.layers[0].get_weights()[0]
word = "king"
word_id = word_index.get(word)
if word_id is not None:
    print(f"Vector for '{word}':\n", embeddings[word_id])
else:
    print(f"'{word}' not found in vocabulary.")
