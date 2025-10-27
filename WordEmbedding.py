from gensim.models import Word2Vec

# Example corpus (list of tokenized sentences)
corpus = [
    ["king", "queen", "man", "woman"],
    ["cat", "dog", "animal", "pet"],
    ["car", "bus", "train", "vehicle"],
    ["apple", "banana", "fruit", "food"],
    ["computer", "keyboard", "mouse", "technology"]
]

# Step 1: Initialize model
model = Word2Vec(vector_size=100, window=1, sg=1, min_count=1)

# Step 2: Build vocabulary
model.build_vocab(corpus)

# Step 3: Train the model
model.train(corpus, total_examples=len(corpus), epochs=30)

# Step 4: Use the model
print("Vector for 'king':")
print(model.wv["king"])

print("\nMost similar to 'king':")
print(model.wv.most_similar("king"))