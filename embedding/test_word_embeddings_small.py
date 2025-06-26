
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'notebook'

EMBEDDINGS_FILE = "test_embeddings.pkl"

def load_test_dictionary():
    # Use a small test set instead of the full dictionary
    words = [
        'red', 'green', 'blue', 'yellow', 'purple', 'orange',
        'man', 'woman', 'child', 'person', 'human',
        'dog', 'cat', 'bird', 'fish', 'animal',
        'tea', 'coffee', 'water', 'juice', 'drink',
        'book', 'table', 'chair', 'house', 'car',
        'happy', 'sad', 'angry', 'excited', 'calm',
        'big', 'small', 'tall', 'short', 'wide',
        'run', 'walk', 'jump', 'sit', 'stand',
        'apple', 'banana', 'orange', 'grape', 'fruit'
    ]
    return words

class EmbeddingWithId:
    def __init__(self, id, embedding):
        self.id = id
        self.embedding = embedding

def generate_and_store_embeddings(all_words):
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("Generating embeddings...")
    embeddings = model.encode(all_words)

    embeddings_with_ids = []
    for word, embedding in zip(all_words, embeddings):
        embedding_obj = EmbeddingWithId(word, embedding)
        embeddings_with_ids.append(embedding_obj)

    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings_with_ids, f)

    return embeddings_with_ids

def query_chroma(embeddings_with_ids, target_words, n_neighbors=20):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    target_embeddings = model.encode(target_words)

    all_similarities = []
    for item in embeddings_with_ids:  # Access the object
        word_similarities = np.dot(item.embedding, target_embeddings.T)
        average_similarity = np.mean(word_similarities)
        all_similarities.append((item.id, average_similarity))

    # Sort by average similarity (descending order)
    all_similarities.sort(key=lambda item: item[1], reverse=True)

    return all_similarities[:n_neighbors]

def load_embeddings():
    try:
        with open(EMBEDDINGS_FILE, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

if __name__ == '__main__':
    print("Starting test with small word list...")
    all_words = load_test_dictionary()
    print(f"Using {len(all_words)} test words")

    model_name = 'all-MiniLM-L6-v2'

    embeddings_with_ids = load_embeddings()
    if embeddings_with_ids is None:
        embeddings_with_ids = generate_and_store_embeddings(all_words)

    # Finding Similar Words
    target_words = ['red', 'coffee', 'book']
    similar_words = query_chroma(embeddings_with_ids, target_words, 10)

    print(f"Top 10 similar words to {', '.join(target_words)}:")
    for word, similarity in similar_words:
        print(f"{word}: {similarity:.4f}")

    print("Test completed successfully!")
