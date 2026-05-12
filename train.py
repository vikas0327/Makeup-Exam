import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt
import pickle

def train_model():
    print("Loading dataset...")
    # 1. Load Dataset
    dataset_path = "dataset.txt"
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = f.read()

    # 2. NLP Preprocessing
    corpus = data.lower().split("\n")
    corpus = [line.strip() for line in corpus if line.strip() != ""]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1

    print(f"Total vocabulary size: {total_words}")

    # Sequence generation
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    # Padding sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    # Display sample sequences
    print(f"\nSample sequences (first 5):\n{input_sequences[:5]}")
    print(f"\nMaximum sequence length: {max_sequence_len}")

    # Create Predictors and Label
    X, labels = input_sequences[:,:-1], input_sequences[:,-1]
    y = tf.keras.utils.to_categorical(labels, num_classes=total_words)

    # 3. LSTM Model Development
    print("\nBuilding LSTM model...")
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(LSTM(150))
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train the model
    print("\nTraining the model (This might take a few minutes)...")
    history = model.fit(X, y, epochs=100, verbose=1)

    # Save the model and tokenizer
    model.save("next_word_model.h5")
    with open("tokenizer.pickle", "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\nModel training complete and saved as 'next_word_model.h5'.")

    # 4. Performance Visualization
    print("\nGenerating performance visualizations...")
    # Plot training accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    # Plot training loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Visualizations saved to 'training_history.png'.")

if __name__ == "__main__":
    # Ensure matplotlib works in headless environments if needed
    import matplotlib
    matplotlib.use('Agg')
    train_model()
