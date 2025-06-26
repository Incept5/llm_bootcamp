import numpy as np
from tabulate import tabulate
import requests
import argparse
import sys
from sklearn.metrics.pairwise import cosine_similarity

# Dictionary of recommended Ollama embedding models with their descriptions
OLLAMA_EMBEDDING_MODELS = {
    "all-minilm": "Default embedding model in Ollama (384 dimensions)",
    "nomic-embed-text": "Nomic AI's text embedding model (768 dimensions)",
    "mxbai-embed-large": "MxbAI's large embedding model (1024 dimensions)",
    "ember": "Ember embedding model (1024 dimensions)",
    "e5": "E5 embedding model (1024 dimensions)",
    "bge": "BGE embedding model (768 dimensions)",
    "gte": "GTE embedding model (768 dimensions)"
}

def get_ollama_embedding(text, model="all-minilm"):
    """
    Get embeddings from Ollama API
    
    Args:
        text (str): Text to embed
        model (str): Ollama model to use for embeddings
        
    Returns:
        list: Embedding vector
    """
    try:
        response = requests.post("http://localhost:11434/api/embeddings",
                                json={"model": model, "prompt": text})
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()["embedding"]
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama server.")
        print("Make sure Ollama is running on http://localhost:11434")
        print("Install Ollama from: https://ollama.com/")
        sys.exit(1)
    except Exception as e:
        print(f"Error getting embedding: {e}")
        if "model" in str(e).lower():
            print(f"Model '{model}' may not be available. Try pulling it with:")
            print(f"  ollama pull {model}")
        sys.exit(1)

def get_ollama_embeddings(sentences, model="all-minilm"):
    """
    Generate embeddings for a list of sentences using Ollama.
    
    Args:
        sentences (list): List of sentences to generate embeddings for
        model (str): Name of the Ollama model to use
        
    Returns:
        np.ndarray: Array of embeddings, one per sentence
    """
    # Check if model exists in our recommended list
    if model not in OLLAMA_EMBEDDING_MODELS:
        print(f"Warning: Using model '{model}' which is not in the recommended list.")
        print("Available recommended models:")
        for m, desc in OLLAMA_EMBEDDING_MODELS.items():
            print(f"  - {m}: {desc}")
    
    print(f"Using Ollama model: {model}")
    
    # Get embeddings for each sentence
    embeddings = []
    for sentence in sentences:
        embedding = get_ollama_embedding(sentence, model)
        embeddings.append(embedding)
    
    return np.array(embeddings)

def calculate_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings.

    Args:
        embedding1 (np.ndarray): First embedding vector
        embedding2 (np.ndarray): Second embedding vector

    Returns:
        float: Cosine similarity score (0-1)
    """
    # Reshape embeddings for sklearn's cosine_similarity
    e1 = embedding1.reshape(1, -1)
    e2 = embedding2.reshape(1, -1)

    # Calculate and return similarity
    return cosine_similarity(e1, e2)[0][0]

def list_models():
    """Print all available recommended Ollama embedding models with descriptions"""
    print("\nAvailable Ollama embedding models:")
    print("-" * 80)
    for model, desc in OLLAMA_EMBEDDING_MODELS.items():
        print(f"{model}")
        print(f"    {desc}")
    print("-" * 80)
    print("\nNote: You may need to pull these models first with 'ollama pull <model>'")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Ollama Embedding Similarity Demo")
    parser.add_argument("--model", type=str, default="all-minilm",
                        help="Ollama model to use for embeddings")
    parser.add_argument("--list-models", action="store_true",
                        help="List all available recommended models and exit")
    args = parser.parse_args()
    
    # If --list-models flag is provided, list models and exit
    if args.list_models:
        list_models()
        return
    
    # Define example sentence pairs
    similar_pair = [
        "The cat sat on the mat.",
        "A feline rested on the rug."
    ]

    dissimilar_pair = [
        "The sky is blue.",
        "Bananas are yellow."
    ]

    mixed_examples = [
        "I enjoy reading science fiction novels.",
        "Reading sci-fi books is my favorite hobby.",
        "The restaurant serves delicious pasta dishes.",
        "My computer needs a hardware upgrade."
    ]

    # Get embeddings for all sentences
    print("Generating Ollama embeddings...")
    all_sentences = similar_pair + dissimilar_pair + mixed_examples
    
    try:
        all_embeddings = get_ollama_embeddings(all_sentences, model=args.model)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Calculate and display similarity for the provided examples
    print("\n--- Example Pairs ---")

    # Similar pair
    sim_score = calculate_similarity(all_embeddings[0], all_embeddings[1])
    print(f"Similar pair similarity score: {sim_score:.4f}")
    print(f"  - \"{similar_pair[0]}\"")
    print(f"  - \"{similar_pair[1]}\"")

    # Dissimilar pair
    dissim_score = calculate_similarity(all_embeddings[2], all_embeddings[3])
    print(f"\nDissimilar pair similarity score: {dissim_score:.4f}")
    print(f"  - \"{dissimilar_pair[0]}\"")
    print(f"  - \"{dissimilar_pair[1]}\"")

    # Compare all mixed examples with each other
    print("\n--- Mixed Examples Similarity Matrix ---")
    start_idx = 4  # Index where mixed examples start in all_embeddings
    n_mixed = len(mixed_examples)

    # Create similarity table
    header = [""] + mixed_examples
    
    # Calculate similarity scores for the table
    table = []
    for i in range(n_mixed):
        row = [mixed_examples[i]]
        for j in range(n_mixed):
            emb_i = all_embeddings[start_idx + i]
            emb_j = all_embeddings[start_idx + j]
            sim = calculate_similarity(emb_i, emb_j)
            row.append(f"{sim:.3f}")
        table.append(row)

    print(tabulate(table, headers=header, tablefmt="grid"))

    # Calculate and print similarity matrix
    for i in range(n_mixed):
        print(f"Sent {i + 1} ", end="")
        for j in range(n_mixed):
            emb_i = all_embeddings[start_idx + i]
            emb_j = all_embeddings[start_idx + j]
            sim = calculate_similarity(emb_i, emb_j)
            print(f"{sim:.4f}    ", end="")
        print()

    print("\nSentence reference:")
    for i, sent in enumerate(mixed_examples):
        print(f"Sentence {i + 1}: {sent}")

if __name__ == "__main__":
    main()
    print("\nTip: Run with --list-models to see all available Ollama embedding models")
    print("Example: python ollama_embedding.py --model nomic-embed-text")