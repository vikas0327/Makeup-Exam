import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

def load_model_and_tokenizer():
    try:
        print("Loading model and tokenizer...")
        model = load_model("next_word_model.h5")
        with open("tokenizer.pickle", "rb") as handle:
            tokenizer = pickle.load(handle)
        # Determine max_sequence_len from the model input shape
        # Input shape is (None, max_sequence_len - 1)
        max_sequence_len = model.input_shape[1] + 1
        return model, tokenizer, max_sequence_len
    except Exception as e:
        print(f"\nError loading model or tokenizer: {e}")
        print("Please run 'python train.py' first to generate the model and tokenizer files.")
        return None, None, None

def predict_next_word(model, tokenizer, max_sequence_len, text, top_k=3):
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    if not token_list:
        return []
        
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    # Predict probabilities for all words
    predicted_probs = model.predict(token_list, verbose=0)[0]
    
    # Get top_k predicted word indices
    top_indices = np.argsort(predicted_probs)[-top_k:][::-1]
    
    # Map indices back to words
    top_words = []
    for index in top_indices:
        for word, idx in tokenizer.word_index.items():
            if idx == index:
                top_words.append((word, predicted_probs[index]))
                break
                
    return top_words

def auto_complete_sentence(model, tokenizer, max_sequence_len, text, num_words=5):
    current_text = text
    for _ in range(num_words):
        top_words = predict_next_word(model, tokenizer, max_sequence_len, current_text, top_k=1)
        if top_words:
            next_word = top_words[0][0]
            current_text += " " + next_word
        else:
            break
    return current_text

def main():
    model, tokenizer, max_sequence_len = load_model_and_tokenizer()
    
    if model is None:
        return
        
    print("\nModel loaded successfully! Type 'exit' to quit.")
    print("-" * 50)
    
    while True:
        user_input = input("\nEnter a sentence fragment: ")
        if user_input.lower() == 'exit':
            break
        if not user_input.strip():
            continue
            
        print("\nPredicting...")
        # Get top 3 words
        top_predictions = predict_next_word(model, tokenizer, max_sequence_len, user_input, top_k=3)
        
        if top_predictions:
            print("\nTop 3 Next Word Predictions:")
            for i, (word, prob) in enumerate(top_predictions, 1):
                print(f"{i}. {word} (Confidence: {prob:.4f})")
            
            print(f"\nPrediction (Highest Confidence): '{top_predictions[0][0]}'")
            
            # Auto-completion
            completed = auto_complete_sentence(model, tokenizer, max_sequence_len, user_input, num_words=3)
            print(f"\nAuto-completion suggestion: {completed}")
        else:
            print("Could not generate a prediction. Please try a different phrase.")

if __name__ == "__main__":
    main()
