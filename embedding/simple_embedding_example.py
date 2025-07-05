import numpy as np
import requests

# mxbai-embed-large 334M, 512 context
# nomic-embed-text 137M, 2k context
# all-minilm 23M, 512 context
# bge-m3 576M, 8k context
# mxbai-embeg-large, 335M, 512 context
# hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF, 596M, 32k context
# hf.co/Qwen/Qwen3-Embedding-4B-GGUF, 4.02B, 32k context

model = "all-minilm"

def get_ollama_embedding(text: str) -> list:
    """Get embedding from Ollama API."""
    data = {
        "model": model,
        "prompt": text
    }
    try:
        response = requests.post("http://localhost:11434/api/embeddings", json=data)
        response.raise_for_status()
        embedding = response.json()["embedding"]
        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_most_similar_sentence(question: str, sentences: list) -> tuple:
    """Find the most similar sentences to the question."""
    # Get embedding for the question
    question_embedding = get_ollama_embedding(question)
    if question_embedding is None:
        return [], None

    # Get the embedding size from the first embedding
    embedding_size = len(question_embedding)

    # Calculate similarities with all sentences
    similarities = []
    for sentence in sentences:
        sentence_embedding = get_ollama_embedding(sentence)
        if sentence_embedding is not None:
            similarity = cosine_similarity(question_embedding, sentence_embedding)
            similarities.append((sentence, similarity))

    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities, embedding_size


def main():
    # Sample sentences to match against
    sentences = [
        "The cat sat on the mat.",
        "What is the weather like today?",
        "Python is a popular programming language.",
        "How do I write extras?",
        "The quick brown fox jumps over the lazy dog.",
        "What's the best way to learn programming?",
        "Machine learning involves training models on data.",
        "Can you help me debug this extras?"
    ]

    while True:
        # Get question from user
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        results, embedding_size = find_most_similar_sentence(question, sentences)

        if embedding_size is None:
            print("Failed to get embeddings.")
            continue

        print(f"\nModel: {model}, size: {embedding_size}, results (sorted by similarity):")
        print("--------------------------------")
        for sentence, similarity in results:
            print(f"Similarity: {similarity:.4f}, Sentence: {sentence}")
            print("--------------------------------")


if __name__ == "__main__":
    main()