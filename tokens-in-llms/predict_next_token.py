import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import matplotlib.pyplot as plt
import threading
import time

# MODEL SELECTION GUIDE:
# - Qwen2.5-7B-Instruct: Best balance of quality vs speed (recommended)
# - Qwen2.5-14B-Instruct: Higher quality but requires ~28GB VRAM
# - Mistral-7B: Alternative high-quality 7B model
# - For GPU with <16GB VRAM, consider using model quantization or smaller models

def load_model():
    torch.set_grad_enabled(False)
    
    # Better model options (choose one):
    # model_path = "Qwen/Qwen2.5-7B-Instruct"      # 7B model - much better than 3B
    # model_path = "Qwen/Qwen2.5-14B-Instruct"     # 14B model - even better but requires more VRAM
    # model_path = "microsoft/DialoGPT-large"       # Good for conversational contexts
    # model_path = "mistralai/Mistral-7B-v0.1"     # Excellent 7B model
    
    model_path = "gpt2-medium"  # Using 7B model for better predictions
    # model_path = "meta-llama/Llama-3.2-3B"  # Requires HF authentication
    
    print(f"Loading model: {model_path}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_safetensors=True)
        
        # Load model with optimized settings
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            use_safetensors=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        print(f"Model loaded successfully. Device: {next(model.parameters()).device}")
        return tokenizer, model
        
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        print("Falling back to smaller model...")
        # Fallback to smaller model if the larger one fails
        fallback_path = "Qwen/Qwen2.5-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(fallback_path, use_safetensors=True)
        model = AutoModelForCausalLM.from_pretrained(fallback_path, use_safetensors=True)
        print(f"Fallback model loaded: {fallback_path}")
        return tokenizer, model

def get_next_token_probs(model, tokenizer, input_string, temperature=1.0, top_p=1.0, top_k=10):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[-1, -1]  # Get logits for the last token

    # Apply temperature
    logits = logits / temperature

    # Compute softmax probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Apply top_k
    if top_k > 0:
        probs, ids = torch.topk(probs, top_k)
    else:
        ids = torch.arange(len(probs))

    # Apply top_p (nucleus sampling)
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs <= top_p
        sorted_probs = sorted_probs[mask]
        # Use sorted_indices to get the correct token ids after filtering
        sorted_token_ids = ids[sorted_indices][mask]
        probs = sorted_probs
        ids = sorted_token_ids

    # Normalize probabilities after filtering
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = torch.ones_like(probs) / len(probs)

    texts = tokenizer.convert_ids_to_tokens(ids)

    return probs, texts

def replace_special_chars(text):
    replacements = {
        'Ġ': ' ', 'Âł': '<LF>', 'Ċ': '<CR>', 'ĉ': '<TAB>', 'Ń': '<NL>', 'Ã¼': "ü",
        'Ā': '', 'Ă': '', '▁': '_', 'âĢ¦': '...', 'âĢľ': '—', 'âĢĺ': '–',
    }
    pattern = '|'.join(map(re.escape, replacements.keys()))
    return re.sub(pattern, lambda m: replacements[m.group()], text)

# Global variables for model and tokenizer
tokenizer = None
model = None
model_loading = False
model_loaded = False

def load_model_async():
    """Load model in background thread"""
    global tokenizer, model, model_loading, model_loaded
    model_loading = True
    try:
        tokenizer, model = load_model()
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
    global model, tokenizer, model_loaded, model_loading
    
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

    probs, texts = get_next_token_probs(model, tokenizer, combined_input, temperature, top_p, top_k)

    result = ""
    for prob, text in zip(probs, texts):
        cleaned_text = replace_special_chars(text)
        if cleaned_text == '<CR>':
            cleaned_text = '\\n'  # Display "\n" in the output
        percentage = prob.item() * 100
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
    global model, tokenizer, model_loaded, model_loading
    
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
    probs, texts = get_next_token_probs(model, tokenizer, new_combined_input, temperature, top_p, top_k)

    result = ""
    for prob, text in zip(probs, texts):
        cleaned_text = replace_special_chars(text)
        if cleaned_text == '<CR>':
            cleaned_text = '\\n'  # Display "\n" in the output
        percentage = prob.item() * 100
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
    gr.Markdown("# Next Token Predictor")
    gr.Markdown("Enter some text, and see the probabilities of the next tokens.")
    
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