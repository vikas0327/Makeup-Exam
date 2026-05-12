import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_from_directory
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__, template_folder='templates', static_folder='static')

# Global variables for model and tokenizer
model = None
tokenizer = None
max_sequence_len = None

def load_ai_assets():
    global model, tokenizer, max_sequence_len
    try:
        model_path = "next_word_model.h5"
        tokenizer_path = "tokenizer.pickle"
        
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            model = load_model(model_path)
            with open(tokenizer_path, "rb") as handle:
                tokenizer = pickle.load(handle)
            max_sequence_len = model.input_shape[1] + 1
            print("Model and Tokenizer loaded successfully!")
        else:
            print("Model files not found. Please train the model first.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.before_request
def before_first_request():
    if model is None:
        load_ai_assets()

def predict_next_word(text, top_k=3):
    if model is None or tokenizer is None:
        return []
        
    token_list = tokenizer.texts_to_sequences([text])[0]
    if not token_list:
        return []
        
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted_probs = model.predict(token_list, verbose=0)[0]
    top_indices = np.argsort(predicted_probs)[-top_k:][::-1]
    
    top_words = []
    for index in top_indices:
        for word, idx in tokenizer.word_index.items():
            if idx == index:
                top_words.append({"word": word, "confidence": float(predicted_probs[index])})
                break
    return top_words

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not trained"}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
        
    text = data['text']
    if not text.strip():
        return jsonify({"predictions": []})
        
    predictions = predict_next_word(text, top_k=3)
    return jsonify({"predictions": predictions})

@app.route('/training_graph')
def training_graph():
    if os.path.exists('training_history.png'):
        return send_from_directory('.', 'training_history.png')
    return "Graph not found", 404

if __name__ == '__main__':
    app.run(debug=True)
