
import gradio as gr
import ollama
from datetime import date
from ollama import Options

# Simple list of zodiac signs
ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", 
    "Leo", "Virgo", "Libra", "Scorpio", 
    "Sagittarius", "Capricorn", "Aquarius", "Pisces"
]

def generate_horoscope(name, star_sign):
    # Get today's date
    today = date.today().strftime('%d-%m-%Y')
    
    # AI system prompt
    system_prompt = """You are an AI astrology assistant called Maude. Provide a short but interesting, positive and
        optimistic horoscope for tomorrow. Provide the response in Markdown format.
        Remember, the user is looking for a positive and optimistic outlook on their future."""

    # User instruction
    instruction = f"Please provide a horoscope for {name} who's star sign is {star_sign}. Today's date is {today}."

    # Call the AI
    response = ollama.chat(
        model="qwen3", 
        think=True, 
        stream=False,
        messages=[
            {'role': 'system', 'content': system_prompt}, 
            {'role': 'user', 'content': instruction}
        ],
        options=Options(temperature=0.8, num_ctx=4096, top_p=0.95, top_k=40, num_predict=-1)
    )

    # Return the horoscope
    return response.message.content

# Create the Gradio interface
interface = gr.Interface(
    fn=generate_horoscope,
    inputs=[
        gr.Textbox(label="Your Name", placeholder="Enter your name"),
        gr.Dropdown(choices=ZODIAC_SIGNS, label="Star Sign", value="Aries")
    ],
    outputs=gr.Markdown(label="Your Horoscope"),
    title="AI Astrology Assistant",
    description="Enter your name and star sign to get a personalized horoscope!"
)

# Launch the interface
if __name__ == "__main__":
    print("Starting AI Astrology Assistant...")
    print("Gradio will automatically find an available port...")
    interface.launch(
        inbrowser=True,  # Automatically open browser
        share=False,     # Don't create public link
        server_name="localhost",  # Run on localhost
        show_api=False   # Don't show API docs
    )
