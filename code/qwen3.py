import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "qwen3:1.7b", "prompt": "Hello", "stream": False}
)

data = response.json()

# Check if there's an error in the response
if "error" in data:
    print(f"Error: {data['error']}")
else:
    print(data["response"])
