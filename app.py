"""
Flask Backend for Legal Chat Assistant
Serves the model and handles API requests
"""

from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)  # Enable CORS for all routes

# Model configuration
MODEL_PATH = "./legal_assistant_model"  # Path to your trained model
MAX_LENGTH = 200
TEMPERATURE = 0.7

# Global variables for model and tokenizer
model = None
tokenizer = None

def load_model():
    """Load the trained model and tokenizer"""
    global model, tokenizer
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        model.eval()  # Set to evaluation mode
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Fallback to base model if custom model not found
        try:
            logger.info("Loading fallback model...")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
            model.eval()
            logger.info("Fallback model loaded")
            return True
        except Exception as e2:
            logger.error(f"Error loading fallback model: {e2}")
            return False

def generate_response(user_input):
    """Generate response from the model"""
    try:
        # Format input
        input_text = f"User: {user_input}\nAssistant:"
        
        # Tokenize
        inputs = tokenizer.encode(input_text, return_tensors='pt', truncation=True, max_length=512)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=len(inputs[0]) + MAX_LENGTH,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
            # Remove any "User:" if it appears in the response
            if "User:" in response:
                response = response.split("User:")[0].strip()
        
        # Add disclaimer
        response += "\n\n*Please note: This is general information only and not legal advice. Consult with a licensed attorney for specific legal matters.*"
        
        return response
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I apologize, but I'm having trouble generating a response. Please try again or rephrase your question."

# Fallback responses for when model is not available
FALLBACK_RESPONSES = {
    "contract": "A contract is a legally binding agreement between parties. For specific contract advice, please consult with a licensed attorney.",
    "divorce": "Divorce laws vary by jurisdiction. I recommend consulting with a family law attorney for guidance specific to your situation.",
    "criminal": "Criminal law matters are serious. If you're facing criminal charges, please seek immediate legal representation.",
    "employment": "Employment law covers workplace rights and obligations. For specific employment issues, consider contacting an employment attorney.",
    "default": "I can provide general legal information, but for specific legal advice, please consult with a licensed attorney in your area."
}

def get_fallback_response(user_input):
    """Get a fallback response based on keywords"""
    user_input_lower = user_input.lower()
    
    for keyword, response in FALLBACK_RESPONSES.items():
        if keyword in user_input_lower:
            return response
    
    return FALLBACK_RESPONSES["default"]

@app.route('/')
def index():
    """Serve the main page"""
    return send_from_directory('', 'index.html')

@app.route('/styles.css')
def styles():
    """Serve CSS file"""
    return send_from_directory('', 'styles.css')

@app.route('/script.js')
def script():
    """Serve JavaScript file"""
    return send_from_directory('', 'script.js')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        data = request.json
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400
        
        # Generate response
        if model and tokenizer:
            response = generate_response(user_input)
        else:
            # Use fallback if model not loaded
            response = get_fallback_response(user_input)
            response += "\n\n*Note: AI model is currently unavailable. This is a simplified response.*"
        
        return jsonify({'response': response})
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model and tokenizer else "not loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status
    })

# Update the JavaScript to call the Flask API
if __name__ == '__main__':
    # Load model on startup
    if load_model():
        logger.info("Model loaded successfully")
    else:
        logger.warning("Running without model - using fallback responses")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)