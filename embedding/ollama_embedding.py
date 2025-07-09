import numpy as np
from tabulate import tabulate
import requests
import time

############################################################################
#
# Some models to try with this, remember to use "ollama pull model" first
#
# hf.co/Qwen/Qwen3-Embedding-4B-GGUF
# hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF
# granite-embedding
# bge-large
# nomic-embed-text
# snowflake-arctic-embed
# mxbai-embed-large
# bge-m3
# snowflake-arctic-embed2
# granite-embedding
#
# Many of the models are quantised or have alternative sizes e.g.
# all-minilm:33m-l12-v2-fp16
# all-minilm:22m-l12-v2-fp16
# all-minilm (defaults to 22m above)
#
############################################################################

def get_available_ollama_models():
    """
    Get list of available models from Ollama API

    Returns:
        list: List of available model names
    """
    try:
        response = requests.get("http://localhost:11434/api/tags")
        response.raise_for_status()
        models_data = response.json()
        return [model['name'] for model in models_data.get('models', [])]
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama server.")
        print("Make sure Ollama is running on http://localhost:11434")
        return []
    except Exception as e:
        print(f"Error getting available models: {e}")
        return []


def filter_embedding_models(available_models):
    """
    Filter available models to only include known embedding models

    Args:
        available_models (list): List of all available models

    Returns:
        list: List of available embedding models
    """
    embedding_models = []
    for model in available_models:
        # Check for common embedding model patterns
        if any(keyword in model.lower() for keyword in ['embed', 'embedding', 'bge', 'gte', 'e5', 'nomic', 'minilm']):
            embedding_models.append(model)

    return embedding_models


def get_ollama_embedding(text, model="all-minilm"):
    """
    Get embeddings from Ollama API

    Args:
        text (str): Text to embed
        model (str): Ollama model to use for embeddings

    Returns:
        list: Embedding vector or None if error
    """
    try:
        response = requests.post("http://localhost:11434/api/embeddings",
                                 json={"model": model, "prompt": text})
        response.raise_for_status()
        return response.json()["embedding"]
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama server.")
        return None
    except Exception as e:
        print(f"Error getting embedding for model '{model}': {e}")
        return None


def get_ollama_embeddings(sentences, model="all-minilm"):
    """
    Generate embeddings for a list of sentences using Ollama.

    Args:
        sentences (list): List of sentences to generate embeddings for
        model (str): Name of the Ollama model to use

    Returns:
        tuple: (embeddings_array, embedding_size) or (None, None) if failed
    """
    print(f"Using Ollama model: {model}")

    # Get embeddings for each sentence
    embeddings = []
    embedding_size = None

    for i, sentence in enumerate(sentences):
        print(f"  Processing sentence {i + 1}/{len(sentences)}...", end="\r")
        embedding = get_ollama_embedding(sentence, model)
        if embedding is None:
            return None, None

        # Convert to numpy array and validate
        embedding_array = np.array(embedding)

        # Get embedding size from first embedding
        if embedding_size is None:
            embedding_size = len(embedding_array)
            print(f"\n  Embedding size: {embedding_size} dimensions")

        is_valid, error_msg = validate_embedding(embedding_array, f"sentence {i + 1}")

        if not is_valid:
            print(f"\n  Error with sentence {i + 1}: {error_msg}")
            print(f"  Sentence: '{sentence[:50]}{'...' if len(sentence) > 50 else ''}'")
            return None, None

        embeddings.append(embedding_array)

    print(f"  Completed all {len(sentences)} sentences.        ")
    return np.array(embeddings), embedding_size


def validate_embedding(embedding, name="embedding"):
    """
    Validate an embedding vector for problematic values

    Args:
        embedding (np.ndarray): Embedding vector to validate
        name (str): Name for error reporting

    Returns:
        tuple: (is_valid, error_message)
    """
    if embedding is None:
        return False, f"{name} is None"

    # Check for NaN or infinite values
    if np.any(np.isnan(embedding)):
        return False, f"{name} contains NaN values"

    if np.any(np.isinf(embedding)):
        return False, f"{name} contains infinite values"

    # Check for zero vector
    if np.allclose(embedding, 0):
        return False, f"{name} is a zero vector"

    # Check for extremely large values that might cause overflow
    if np.any(np.abs(embedding) > 1e10):
        return False, f"{name} contains extremely large values"

    return True, "Valid"


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def calculate_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two embeddings with validation.

    Args:
        embedding1 (np.ndarray): First embedding vector
        embedding2 (np.ndarray): Second embedding vector

    Returns:
        float or None: Cosine similarity score (-1 to 1) or None if invalid
    """
    # Validate embeddings
    valid1, msg1 = validate_embedding(embedding1, "embedding1")
    valid2, msg2 = validate_embedding(embedding2, "embedding2")

    if not valid1:
        print(f"Warning: {msg1}")
        return None

    if not valid2:
        print(f"Warning: {msg2}")
        return None

    try:
        # Calculate similarity using simple numpy implementation
        similarity = cosine_similarity(embedding1, embedding2)

        # Check if result is valid
        if np.isnan(similarity) or np.isinf(similarity):
            print(f"Warning: Cosine similarity calculation resulted in {similarity}")
            return None

        return similarity

    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return None


def run_similarity_analysis(model_name, all_embeddings, embedding_size, similar_pair, dissimilar_pair, mixed_examples):
    """
    Run similarity analysis for a given model and its embeddings

    Args:
        model_name (str): Name of the model
        all_embeddings (np.ndarray): Precomputed embeddings
        embedding_size (int): Size of the embedding vectors
        similar_pair (list): Pair of similar sentences
        dissimilar_pair (list): Pair of dissimilar sentences
        mixed_examples (list): List of mixed example sentences

    Returns:
        dict or None: Results dictionary or None if analysis failed
    """
    print(f"\n{'=' * 80}")
    print(f"RESULTS FOR MODEL: {model_name.upper()}")
    print(f"Embedding size: {embedding_size} dimensions")
    print(f"{'=' * 80}")

    # Calculate and display similarity for the provided examples
    print("\n--- Example Pairs ---")

    # Similar pair
    sim_score = calculate_similarity(all_embeddings[0], all_embeddings[1])
    if sim_score is None:
        print("ERROR: Could not calculate similarity for similar pair")
        return None

    print(f"Similar pair similarity score: {sim_score:.4f}")
    print(f"  - \"{similar_pair[0]}\"")
    print(f"  - \"{similar_pair[1]}\"")

    # Dissimilar pair
    dissim_score = calculate_similarity(all_embeddings[2], all_embeddings[3])
    if dissim_score is None:
        print("ERROR: Could not calculate similarity for dissimilar pair")
        return None

    print(f"\nDissimilar pair similarity score: {dissim_score:.4f}")
    print(f"  - \"{dissimilar_pair[0]}\"")
    print(f"  - \"{dissimilar_pair[1]}\"")

    # Compare all mixed examples with each other
    print("\n--- Mixed Examples Similarity Matrix ---")
    start_idx = 4  # Index where mixed examples start in all_embeddings
    n_mixed = len(mixed_examples)

    # Create similarity table
    header = ["Sentence"] + [f"S{i + 1}" for i in range(n_mixed)]

    # Calculate similarity scores for the table
    table = []

    for i in range(n_mixed):
        row = [f"S{i + 1}"]
        for j in range(n_mixed):
            emb_i = all_embeddings[start_idx + i]
            emb_j = all_embeddings[start_idx + j]
            sim = calculate_similarity(emb_i, emb_j)
            if sim is None:
                row.append("ERROR")
            else:
                row.append(f"{sim:.3f}")
        table.append(row)

    print(tabulate(table, headers=header, tablefmt="grid"))

    # Print sentence reference
    print("\nSentence reference:")
    for i, sent in enumerate(mixed_examples):
        print(f"S{i + 1}: {sent}")

    # Return results for summary
    return {
        'similar_score': sim_score,
        'dissimilar_score': dissim_score,
        'difference': sim_score - dissim_score,
        'embedding_size': embedding_size
    }


def main():
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

    all_sentences = similar_pair + dissimilar_pair + mixed_examples

    # Get all available embedding models
    available_models = get_available_ollama_models()
    if not available_models:
        print("No models available. Make sure Ollama is running and has models installed.")
        return

    embedding_models = filter_embedding_models(available_models)
    if not embedding_models:
        print("No embedding models found among available models.")
        print("Available models:", available_models)
        print("Consider installing embedding models like: ollama pull all-minilm")
        return

    print(f"Found {len(embedding_models)} embedding model(s): {', '.join(embedding_models)}")

    # First, run demo with all-minilm (or first available if all-minilm not available)
    default_model = "all-minilm" if "all-minilm" in embedding_models else embedding_models[0]
    if default_model != "all-minilm":
        print(f"all-minilm not available, using {default_model} for initial demo")

    print(f"\n{'*' * 60}")
    print(f"INITIAL DEMO WITH MODEL: {default_model}")
    print(f"{'*' * 60}")

    # Generate embeddings for initial demo
    print("Generating Ollama embeddings...")
    try:
        demo_embeddings = get_ollama_embeddings(all_sentences, model=default_model)
        if demo_embeddings is not None:
            run_similarity_analysis(default_model, demo_embeddings, similar_pair, dissimilar_pair, mixed_examples)
    except Exception as e:
        print(f"Error in initial demo with {default_model}: {e}")

    # Now iterate through ALL available embedding models
    print(f"\n{'=' * 80}")
    print("COMPARISON OF ALL AVAILABLE EMBEDDING MODELS")
    print(f"{'=' * 80}")

    # Results storage for comparison
    results_summary = []

    # Test each available embedding model
    for model_name in embedding_models:
        print(f"\n{'*' * 60}")
        print(f"TESTING MODEL: {model_name}")
        print(f"{'*' * 60}")

        start_time = time.time()

        # Generate embeddings
        print("Generating Ollama embeddings...")
        try:
            all_embeddings, embedding_size = get_ollama_embeddings(all_sentences, model=model_name)
            if all_embeddings is None:
                print(f"Failed to generate embeddings for model: {model_name}")
                continue

            # Run similarity analysis
            analysis_result = run_similarity_analysis(model_name, all_embeddings, embedding_size, similar_pair,
                                                      dissimilar_pair, mixed_examples)

            if analysis_result is None:
                print(f"Similarity analysis failed for model: {model_name}")
                continue

            # Store results for summary
            results_summary.append({
                'model': model_name,
                'similar_score': analysis_result['similar_score'],
                'dissimilar_score': analysis_result['dissimilar_score'],
                'difference': analysis_result['difference'],
                'processing_time': time.time() - start_time,
                'embedding_size': analysis_result['embedding_size']
            })

        except Exception as e:
            print(f"Error processing model {model_name}: {e}")
            continue

    # Print summary comparison if multiple models were tested
    if len(results_summary) > 1:
        print(f"\n{'=' * 80}")
        print("SUMMARY COMPARISON OF ALL MODELS")
        print(f"{'=' * 80}")

        # Create summary table with right-aligned embedding size column
        summary_headers = ["Model", "Embedding Size", "Similar Score", "Dissimilar Score", "Difference", "Time (s)"]
        summary_table = []

        for result in results_summary:
            summary_table.append([
                result['model'],
                f"{result['embedding_size']:,}",
                f"{result['similar_score']:.4f}",
                f"{result['dissimilar_score']:.4f}",
                f"{result['difference']:.4f}",
                f"{result['processing_time']:.2f}"
            ])

        # Use colalign to right-align the embedding size column
        print(tabulate(summary_table, headers=summary_headers, tablefmt="grid",
                      colalign=("left", "right", "center", "center", "center", "center")))

        # Show information about failed models
        total_models = len(embedding_models)
        successful_models = len(results_summary)
        failed_models = total_models - successful_models

        if failed_models > 0:
            print(f"\nNote: {failed_models} model(s) failed to produce valid embeddings")


if __name__ == "__main__":
    main()
    print("\nTip: Install embedding models with 'ollama pull all-minilm' or 'ollama pull nomic-embed-text' (see code)")