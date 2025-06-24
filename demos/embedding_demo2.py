import numpy as np
from tabulate import tabulate
import requests

def get_ollama_embedding(text):
    """Get embedding from Ollama API for the given text.
    
    Available models:
    - "nomic-embed-text" (768 dimensions)
    - "all-minilm:33m-l12-v2-fp16" (384 dimensions)
    """
    try:
        response = requests.post("http://localhost:11434/api/embeddings",
                                 json={"model": "nomic-embed-text", "prompt": text})
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()["embedding"]
    except requests.exceptions.RequestException as e:
        print(f"Error getting embedding for '{text}': {e}")
        raise
    except KeyError as e:
        print(f"Unexpected API response format: {e}")
        print(f"Response: {response.json()}")
        raise

def main():
    words = ["beer", "wine", "coffee", "espresso"]
    
    print("Word Embedding Similarity Demo")
    print("==============================")
    print(f"Computing similarities for: {', '.join(words)}")
    print("Using Ollama's nomic-embed-text model...\n")

    # Get embeddings and normalize them
    embeddings = np.array([get_ollama_embedding(word) for word in words])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]

    # Calculate similarities using dot product of normalized vectors
    similarities = np.dot(embeddings, embeddings.T)

    # Create similarity table
    header = [""] + words
    table = [[words[i]] + [f"{similarities[i][j]:.3f}"
                           for j in range(len(words))] for i in range(len(words))]

    print("Cosine Similarity Matrix:")
    print("(1.0 = identical meaning, 0.0 = no similarity, -1.0 = opposite meaning)")
    print()
    print(tabulate(table, headers=header, tablefmt="grid"))
    print()
    print("Notice how coffee and espresso have high similarity (0.695),")
    print("and beer and wine are also related (0.633).")

if __name__ == "__main__":
    main()
