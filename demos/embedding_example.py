import numpy as np
import requests


def get_ollama_embedding(text: str) -> list:
    """Get embedding from Ollama API."""
    data = {
        "model": "all-minilm:33m-l12-v2-fp16",
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


def find_most_similar_sentence(question: str, sentences: list) -> list:
    """Find the most similar sentences to the question."""
    # Get embedding for the question
    question_embedding = get_ollama_embedding(question)
    if question_embedding is None:
        return []

    # Calculate similarities with all sentences
    similarities = []
    for sentence in sentences:
        sentence_embedding = get_ollama_embedding(sentence)
        if sentence_embedding is not None:
            similarity = cosine_similarity(question_embedding, sentence_embedding)
            similarities.append((sentence, similarity))

    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities


def main():
    # Sample sentences to match against
    sentences = [
        "The cat sat on the mat.",
        "What is the weather like today?",
        "Python is a popular programming language.",
        "How do I write code?",
        "The quick brown fox jumps over the lazy dog.",
        "What's the best way to learn programming?",
        "Machine learning involves training models on data.",
        "Can you help me debug this code?"
    ]

    while True:
        # Get question from user
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        print("\nFinding similar sentences...")
        results = find_most_similar_sentence(question, sentences)

        print("\nResults (sorted by similarity):")
        print("--------------------------------")
        for sentence, similarity in results:
            print(f"Similarity: {similarity:.4f}")
            print(f"Sentence: {sentence}")
            print("--------------------------------")


if __name__ == "__main__":
    print("Sentence Similarity Matcher")
    print("Make sure Ollama is running locally on port 11434")
    main()