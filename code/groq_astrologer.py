import os
from groq import Groq
from datetime import date
from rich.console import Console
from rich.markdown import Markdown

client = Groq( api_key=os.environ.get("GROQ_API_KEY"))

def get_user_input():
    name = input("Enter your name: ")
    star_sign = input("Enter your star sign: ")
    return name, star_sign

def main():
    today = date.today().strftime('%d-%m-%Y')
    name, star_sign = get_user_input()

    system_prompt = f"""You are an AI astrology assistant called Maude. You will provide an interesting, positive and
        optimistic horoscope for the near future. End with a general outlook for the future.
        Provide the response in Markdown format.
        Please use British English spelling and grammar."""
    instruction = f"Please provide a horoscope for {name} who's star sign is {star_sign}. Today's date is {today}."

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": instruction,
            }
        ],
        model="llama-3.1-70b-versatile",
    )
    console = Console()

    # Render the Markdown content
    markdown = Markdown(chat_completion.choices[0].message.content)
    console.print(markdown)

if __name__ == "__main__":
    main()