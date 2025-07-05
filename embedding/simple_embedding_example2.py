import numpy as np
from tabulate import tabulate
import requests

# mxbai-embed-large 334M, 512 context
# nomic-embed-text 137M, 2k context
# all-minilm 23M, 512 context
# bge-m3 576M, 8k context
# mxbai-embeg-large, 335M, 512 context
# hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF, 596M, 32k context
# hf.co/Qwen/Qwen3-Embedding-4B-GGUF, 4.02B, 32k context

model = "all-minilm"

def get_ollama_embedding(text):
    """Get embedding from Ollama API for the given text."""
    try:
        response = requests.post("http://localhost:11434/api/embeddings",
                                 json={"model": model, "prompt": text})
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
    words = ["beer", "cerveza", "wine", "vin", "coffee", "espresso"]

    print("Word Embedding Similarity Demo")
    print("==============================")
    print(f"Computing similarities for: {', '.join(words)}")

    # Get embeddings and normalize them
    embeddings = np.array([get_ollama_embedding(word) for word in words])

    # Get embedding size from the first embedding
    embedding_size = len(embeddings[0])

    print(f"Model: {model}, size: {embedding_size}")
    print()

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]

    # Calculate similarities using dot product of normalized vectors
    similarities = np.dot(embeddings, embeddings.T)

    # Create similarity table
    header = [""] + words
    table = [[words[i]] + [f"{similarities[i][j]:.3f}"
                           for j in range(len(words))] for i in range(len(words))]

    print("Cosine Similarity Matrix:")
    print("1.0 = identical meaning, 0.0 = no similarity, -1.0 = opposite meaning):")
    print()
    print(tabulate(table, headers=header, tablefmt="grid"))
    print()

if __name__ == "__main__":
    main()