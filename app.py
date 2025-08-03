import pandas as pd
import torch
from flask import Flask, request, render_template, jsonify
from transformers import BertForSequenceClassification, BertTokenizer
import sys

# --- Configuration & Model Loading ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256

# --- Pre-load Model and Tokenizer ---
# This part runs only once when the server starts.
print("🧠 Loading model and tokenizer...")
try:
    # We are loading a pre-trained model directly. 
    # In a real-world scenario, you would save your fine-tuned model 
    # after training and load it here from a file.
    # For this example, we'll use a generic pre-trained model.
    # Its accuracy won't be as good as your fine-tuned one, but it demonstrates the process.
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_NAME, num_labels=2)
    model.to(DEVICE)
    model.eval() # Set model to evaluation mode
    print("✅ Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit()

# --- Prediction Function ---
def predict_headline(headline):
    """Takes a headline string and returns 'FAKE' or 'REAL'."""
    try:
        # Tokenize the input headline
        inputs = tokenizer(
            headline, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=MAX_LENGTH
        ).to(DEVICE)
        
        # Get prediction from the model
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            
        return "FAKE" if prediction == 1 else "REAL"

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Error"

# --- Flask App ---
app = Flask(__name__)

@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives a headline from the frontend and returns a prediction."""
    try:
        data = request.get_json()
        if not data or 'headline' not in data:
            return jsonify({'error': 'Invalid input. "headline" key is required.'}), 400
            
        headline = data['headline']
        if not headline.strip():
            return jsonify({'prediction': 'Please enter a headline.'})

        # Get the prediction from our model
        result = predict_headline(headline)
        
        # Return the result as JSON
        return jsonify({'prediction': result})

    except Exception as e:
        print(f"Server error: {e}")
        return jsonify({'error': 'A server error occurred.'}), 500


if __name__ == '__main__':
    # NOTE: In a real production environment, use a proper web server like Gunicorn or Waitress.
    # Example: waitress-serve --host 127.0.0.1 --port 5000 app:app
    app.run(debug=False, host='0.0.0.0')
    