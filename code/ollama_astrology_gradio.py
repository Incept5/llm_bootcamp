
"""
AI Astrology Assistant with Gradio Web Interface
A beginner-friendly web interface for generating personalized horoscopes using Ollama.
"""

import gradio as gr
import ollama
from datetime import date
from ollama import Options

# Configuration
LLM = "qwen3"  # The AI model to use
THINKING = True  # Enable AI thinking process display

# List of all zodiac signs for the dropdown
ZODIAC_SIGNS = [
    "♈ Aries (March 21 - April 19)",
    "♉ Taurus (April 20 - May 20)", 
    "♊ Gemini (May 21 - June 20)",
    "♋ Cancer (June 21 - July 22)",
    "♌ Leo (July 23 - August 22)",
    "♍ Virgo (August 23 - September 22)",
    "♎ Libra (September 23 - October 22)",
    "♏ Scorpio (October 23 - November 21)",
    "♐ Sagittarius (November 22 - December 21)",
    "♑ Capricorn (December 22 - January 19)",
    "♒ Aquarius (January 20 - February 18)",
    "♓ Pisces (February 19 - March 20)"
]

def generate_horoscope(name, star_sign_full):
    """
    Generate a personalized horoscope using AI.
    
    Args:
        name (str): User's name
        star_sign_full (str): Full star sign text from dropdown
    
    Returns:
        tuple: (horoscope_text, thinking_process)
    """
    try:
        # Extract just the star sign name from the dropdown selection
        # Example: "♈ Aries (March 21 - April 19)" -> "Aries"
        star_sign = star_sign_full.split(' ')[1]
        
        # Get today's date
        today = date.today().strftime('%d-%m-%Y')
        
        # System prompt - tells the AI how to behave
        system_prompt = """You are an AI astrology assistant called Maude. Provide a short but interesting, positive and
            optimistic horoscope for tomorrow. Provide the response in Markdown format.
            Remember, the user is looking for a positive and optimistic outlook on their future."""

        # User instruction - what we want the AI to do
        instruction = f"Please provide a horoscope for {name} who's star sign is {star_sign}. Today's date is {today}."

        # Call the AI model
        response = ollama.chat(
            model=LLM, 
            think=THINKING, 
            stream=False,
            messages=[
                {'role': 'system', 'content': system_prompt}, 
                {'role': 'user', 'content': instruction}
            ],
            options=Options(
                temperature=0.8,    # Controls randomness (0.0 = deterministic, 1.0 = very random)
                num_ctx=4096,      # Context window size
                top_p=0.95,        # Nucleus sampling parameter
                top_k=40,          # Top-k sampling parameter
                num_predict=-1     # Maximum tokens to generate (-1 = unlimited)
            )
        )

        # Extract the horoscope text
        horoscope = response.message.content
        
        # Extract thinking process if available
        thinking_process = ""
        if hasattr(response.message, 'thinking') and response.message.thinking:
            thinking_process = f"🤔 **Maude's Thinking Process:**\n\n{response.message.thinking}"
        else:
            thinking_process = "Thinking process not available for this response."

        return horoscope, thinking_process

    except Exception as e:
        # Handle errors gracefully
        error_message = f"Sorry, I encountered an error: {str(e)}\n\nPlease make sure Ollama is running and the '{LLM}' model is installed."
        return error_message, "Error occurred - no thinking process available."

def create_interface():
    """
    Create and configure the Gradio web interface.
    """
    
    # Create the interface
    interface = gr.Interface(
        fn=generate_horoscope,  # Function to call when user submits
        
        # Input components
        inputs=[
            gr.Textbox(
                label="Your Name",
                placeholder="Enter your name here...",
                value="",
                lines=1
            ),
            gr.Dropdown(
                choices=ZODIAC_SIGNS,
                label="Select Your Star Sign",
                value=ZODIAC_SIGNS[0]  # Default to first option (Aries)
            )
        ],
        
        # Output components
        outputs=[
            gr.Markdown(
                label="✨ Your Personal Horoscope",
                show_label=True
            ),
            gr.Accordion(
                gr.Markdown(),
                label="🤔 AI Thinking Process (Click to expand)",
                open=False  # Collapsed by default
            )
        ],
        
        # Interface configuration
        title="🔮 AI Astrology Assistant - Maude",
        description="""
        Welcome to Maude, your personal AI astrology assistant! 
        
        Simply enter your name, select your star sign, and get a personalized, optimistic horoscope for tomorrow.
        
        **How it works:**
        1. Enter your name in the text field
        2. Select your zodiac sign from the dropdown
        3. Click 'Submit' to generate your horoscope
        4. Expand the 'AI Thinking Process' section to see how Maude created your reading
        
        *Powered by Ollama and the Qwen3 AI model*
        """,
        
        # Styling and behavior
        theme=gr.themes.Soft(),  # Use a soft, friendly theme
        submit_btn="Generate My Horoscope 🔮",
        clear_btn="Clear Form",
        
        # Examples to help users get started
        examples=[
            ["Alice", "♌ Leo (July 23 - August 22)"],
            ["Bob", "♏ Scorpio (October 23 - November 21)"],
            ["Emma", "♓ Pisces (February 19 - March 20)"]
        ],
        
        # Additional configuration
        cache_examples=False,  # Don't cache examples (generate fresh each time)
        show_api=False,       # Hide API documentation
        allow_flagging="never"  # Disable flagging feature
    )
    
    return interface

def main():
    """
    Main function to launch the Gradio interface.
    """
    print("🔮 Starting AI Astrology Assistant...")
    print("📝 Make sure Ollama is running with the 'qwen3' model installed!")
    print("🌐 The web interface will open in your browser...")
    
    # Create and launch the interface
    interface = create_interface()
    
    # Launch with specific settings
    interface.launch(
        share=False,      # Set to True if you want to create a public link
        server_port=7860, # Port number (change if needed)
        debug=False,      # Set to True for debugging
        show_error=True   # Show errors in the interface
    )

if __name__ == "__main__":
    main()
