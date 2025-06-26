import ollama
import json

def generate_formatted_response(prompt):
    try:
        response = ollama.generate(
            model="qwen2.5-coder",
            prompt=prompt,
            format="json",  # Only accepts "" or "json"
            options={ "num_ctx": 8192,"temperature": 0.3 }
        )
        return response['response']
    except Exception as e:
        print("Error:", e)
        return None

def main():
    prompt = """List the numbers from 1 to 10 and their names in English, French, German, Chinese, Russian and Arabic.
    Provide the output in this exact JSON format:
    {
      "numbers": [
        {
          "number": 1,
          "English": "one",
          "French": "un",
          "German": "ein"
          "Chinese": "一"
          "Russian": "один"
          "Arabic": "واحد"
        },
        ...and so on for numbers 1-10
      ]
    }"""
    response = generate_formatted_response(prompt)

    try:
        if response:
            parsed = json.loads(response)
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print("Received non-JSON response:")
        print(response)

if __name__ == "__main__":
    main()