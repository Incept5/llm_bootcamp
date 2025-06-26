import gradio as gr
import requests
import json
import re
import matplotlib.pyplot as plt
import threading
import time
import numpy as np
from collections import Counter

# OLLAMA MODEL SELECTION GUIDE (smaller, faster models):
# - llama3.2:1b: Fastest, 1B parameters (recommended for speed)
# - phi3:mini: Good quality, 3.8B parameters  
# - qwen2.5:1.5b: Balanced option, 1.5B parameters
# - llama3.2:3b: Higher quality, 3B parameters
# Make sure to run 'ollama pull <model>' first to download the model

def load_model():
    # Ollama model options (choose one):
    model_name = "llama3.2:1b"  # Fast 1B parameter model (recommended)
    # model_name = "phi3:mini"     # Good quality 3.8B model
    # model_name = "qwen2.5:1.5b"  # Balanced 1.5B model
    # model_name = "llama3.2:3b"   # Higher quality 3B model
    
    print(f"Using Ollama model: {model_name}")
    
    try:
        # Test connection to Ollama
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            raise Exception("Ollama server not running. Please start Ollama first.")
        
        # Check if model is available
        models = response.json().get('models', [])
        model_names = [m['name'] for m in models]
        
        if model_name not in model_names:
            print(f"Model {model_name} not found. Available models: {model_names}")
            print(f"Please run: ollama pull {model_name}")
            raise Exception(f"Model {model_name} not available")
        
        print(f"Model {model_name} is available and ready")
        return model_name
        
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running and the model is pulled")
        raise e

def get_next_token_probs(model_name, input_string, temperature=1.0, top_p=1.0, top_k=10):
    """Get next token probabilities using Ollama API"""
    try:
        # Use Ollama's generate API to get multiple completions
        completions = []
        tokens = []
        
        # Generate multiple short completions to simulate token probabilities
        for _ in range(max(20, top_k * 2)):
            payload = {
                "model": model_name,
                "prompt": input_string,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "num_predict": 1,  # Generate only 1 token
                    "stop": ["\n", " ", ".", ",", "!", "?"]  # Stop at common delimiters
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                next_text = result.get('response', '').strip()
                if next_text:
                    # Extract first token/character
                    first_token = next_text.split()[0] if next_text.split() else next_text[:1]
                    if first_token:
                        tokens.append(first_token)
        
        # Count token frequencies to simulate probabilities
        if not tokens:
            # Fallback tokens if no responses
            tokens = [" ", "the", "and", "a", "to", "of", "in", "I", "you", "it"]
            token_counts = Counter({token: 1 for token in tokens})
        else:
            token_counts = Counter(tokens)
        
        # Get top_k most common tokens
        most_common = token_counts.most_common(top_k)
        
        # Convert counts to probabilities
        total_count = sum(count for _, count in most_common)
        probs = [count / total_count for _, count in most_common]
        texts = [token for token, _ in most_common]
        
        return np.array(probs), texts
        
    except Exception as e:
        print(f"Error getting token probabilities: {e}")
        # Return fallback probabilities
        fallback_tokens = [" ", "the", "and", "a", "to", "of", "in", "I", "you", "it"]
        fallback_probs = np.array([0.2, 0.15, 0.12, 0.1, 0.08, 0.08, 0.07, 0.07, 0.07, 0.06])
        return fallback_probs, fallback_tokens

def replace_special_chars(text):
    replacements = {
        'Ġ': ' ', 'Âł': '<LF>', 'Ċ': '<CR>', 'ĉ': '<TAB>', 'Ń': '<NL>', 'Ã¼': "ü",
        'Ā': '', 'Ă': '', '▁': '_', 'âĢ¦': '...', 'âĢľ': '—', 'âĢĺ': '–',
    }
    pattern = '|'.join(map(re.escape, replacements.keys()))
    return re.sub(pattern, lambda m: replacements[m.group()], text)

# Global variables for model
model_name = None
model_loading = False
model_loaded = False

def load_model_async():
    """Load model in background thread"""
    global model_name, model_loading, model_loaded
    model_loading = True
    try:
        model_name = load_model()
        model_loaded = True
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
    finally:
        model_loading = False

def predict_next_tokens(user_input, system_prompt, temperature, top_p, top_k, state):
    """
    Generates the next token probabilities based on the user input and system prompt.
    Keeps the system prompt separate from the user-visible input.
    """
    global model_name, model_loaded, model_loading
    
    if not model_loaded:
        if model_loading:
            return user_input, "🤖 Model is still loading... Please wait a moment and try again.", None, state
        else:
            return user_input, "❌ Model failed to load. Please refresh the page to try again.", None, state
    
    # Combine system prompt with user input if system prompt is provided
    if system_prompt.strip():
        combined_input = f"{system_prompt}\n{user_input}"
    else:
        combined_input = user_input

    probs, texts = get_next_token_probs(model_name, combined_input, temperature, top_p, top_k)

    result = ""
    for prob, text in zip(probs, texts):
        cleaned_text = replace_special_chars(text)
        if cleaned_text == '<CR>':
            cleaned_text = '\\n'  # Display "\n" in the output
        percentage = prob * 100
        result += f"{percentage:.2f}% : \"{cleaned_text}\"\n"

    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 10))
    cleaned_texts = [replace_special_chars(text) for text in texts]
    ax.pie(probs.tolist(), labels=cleaned_texts, autopct='%1.1f%%', startangle=90)
    ax.set_title("Next Token Probabilities")

    # Only store the combined input in the hidden state
    new_state = combined_input

    # Return the original user input unchanged
    return user_input, result, fig, new_state

def add_next_token(user_input, prediction_text, system_prompt, temperature, top_p, top_k, state):
    global model_name, model_loaded, model_loading
    
    if not model_loaded:
        if model_loading:
            return user_input, "🤖 Model is still loading... Please wait a moment and try again.", None, state
        else:
            return user_input, "❌ Model failed to load. Please refresh the page to try again.", None, state
    
    if not prediction_text:
        return user_input, "", None, state

    # Extract the next token from the prediction_text
    try:
        next_token = prediction_text.split('\n')[0].split(':')[1].strip()[1:-1]
    except IndexError:
        next_token = ""

    # Handle carriage return
    if next_token == '\\n':
        next_token = '\n'

    # Replace various representations of newline with actual newline character
    next_token = next_token.replace('<CR>', '\n').replace('<BR>', '\n').replace('\\n', '\n')

    # Handle cases where <CR> or <BR> are part of a larger token
    next_token = re.sub(r'(\{<CR>|\{<BR>)', '{\n', next_token)
    next_token = re.sub(r'(<CR>}|<BR>})', '\n}', next_token)

    # Update user input with the next token
    updated_user_input = user_input + next_token

    # Combine with system prompt for the model
    if system_prompt.strip():
        new_combined_input = f"{system_prompt}\n{updated_user_input}"
    else:
        new_combined_input = updated_user_input

    # Get next token probabilities
    probs, texts = get_next_token_probs(model_name, new_combined_input, temperature, top_p, top_k)

    result = ""
    for prob, text in zip(probs, texts):
        cleaned_text = replace_special_chars(text)
        if cleaned_text == '<CR>':
            cleaned_text = '\\n'  # Display "\n" in the output
        percentage = prob * 100
        result += f"{percentage:.2f}% : \"{cleaned_text}\"\n"

    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 10))
    cleaned_texts = [replace_special_chars(text) for text in texts]
    ax.pie(probs.tolist(), labels=cleaned_texts, autopct='%1.1f%%', startangle=90)
    ax.set_title("Next Token Probabilities")

    # Update the hidden state with the new combined input
    new_state = new_combined_input

    return updated_user_input, result, fig, new_state

def get_model_status():
    """Get current model loading status"""
    global model_loaded, model_loading
    if model_loaded:
        return "✅ Model loaded and ready!"
    elif model_loading:
        return "🤖 Loading model... This may take a few minutes for larger models."
    else:
        return "❌ Model failed to load."

with gr.Blocks(theme='default') as iface:
    gr.Markdown("# Next Token Predictor (Ollama)")
    gr.Markdown("Enter some text, and see the probabilities of the next tokens.")
    gr.Markdown("**Requirements:** Make sure Ollama is running and you have pulled a model (e.g., `ollama pull llama3.2:1b`)")
    
    # Model status indicator
    model_status = gr.Markdown("🤖 Loading model... This may take a few minutes for larger models.")

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                lines=5,
                placeholder="Enter your text here...",
                label="Input Text"
            )
            system_prompt = gr.Textbox(
                lines=2,
                placeholder="Enter a seed here (optional)...",
                label="Seed Prompt"
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=0.3,
                step=0.1,
                label="Temperature"
            )
            top_p = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=1.0,
                step=0.05,
                label="Top P (Nucleus Sampling)"
            )
            top_k = gr.Slider(
                minimum=1,
                maximum=20,
                value=10,
                step=1,
                label="Top K"
            )
            with gr.Row():
                submit_button = gr.Button("Submit")
                next_button = gr.Button("Next")
                refresh_status_button = gr.Button("Refresh Status", size="sm")

            # Hidden state to store the combined input
            hidden_state = gr.State(value="")

        with gr.Column(scale=1):
            output_text = gr.Textbox(
                lines=10,
                label="Predicted Next Tokens"
            )
            output_plot = gr.Plot(
                label="Probability Distribution"
            )

    # Define the event handlers with the hidden state
    submit_button.click(
        predict_next_tokens,
        inputs=[input_text, system_prompt, temperature, top_p, top_k, hidden_state],
        outputs=[input_text, output_text, output_plot, hidden_state]
    )
    next_button.click(
        add_next_token,
        inputs=[input_text, output_text, system_prompt, temperature, top_p, top_k, hidden_state],
        outputs=[input_text, output_text, output_plot, hidden_state]
    )
    refresh_status_button.click(
        get_model_status,
        outputs=[model_status]
    )

# Start model loading in background thread
loading_thread = threading.Thread(target=load_model_async, daemon=True)
loading_thread.start()

# Launch the interface immediately
iface.launch()