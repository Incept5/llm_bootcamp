import numpy as np
import argparse
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

# Dictionary of recommended Hugging Face embedding models with their descriptions
HF_EMBEDDING_MODELS = {
    "bert-base-uncased": "Original BERT base model (768 dimensions)",
    "roberta-base": "RoBERTa base model with improved training (768 dimensions)",
    "distilbert-base-uncased": "Distilled version of BERT, smaller and faster (768 dimensions)",
    "albert-base-v2": "A Lite BERT with parameter reduction techniques (768 dimensions)",
    "xlm-roberta-base": "Multilingual RoBERTa model supporting 100 languages (768 dimensions)",
    "microsoft/mpnet-base": "MPNet with better performance than BERT/RoBERTa (768 dimensions)",
    "google/electra-small-discriminator": "Smaller, efficient ELECTRA model (256 dimensions)",
    "sentence-transformers/all-MiniLM-L6-v2": "Optimized for sentence embeddings (384 dimensions)",
    "intfloat/e5-small-v2": "E5 model optimized for text embeddings (384 dimensions)",
    "facebook/contriever-msmarco": "Contriever model fine-tuned on MS MARCO (768 dimensions)"
}

def mean_pooling(model_output, attention_mask):
    """
    Mean pooling to get sentence embeddings from token embeddings
    """
    token_embeddings = model_output[0]  # First element of model_output contains token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_huggingface_embeddings(sentences, model_name="bert-base-uncased"):
    """
    Generate embeddings for a list of sentences using Hugging Face models.
    
    Args:
        sentences (list): List of sentences to generate embeddings for
        model_name (str): Name of the Hugging Face model to use
        
    Returns:
        np.ndarray: Array of embeddings, one per sentence
    """
    # Check if model exists in our recommended list
    if model_name not in HF_EMBEDDING_MODELS and not model_name.startswith("custom:"):
        print(f"Warning: Using model '{model_name}' which is not in the recommended list.")
        print("Available recommended models:")
        for model, desc in HF_EMBEDDING_MODELS.items():
            print(f"  - {model}: {desc}")
    
    # Load tokenizer and model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
    
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Perform mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    # Convert to numpy and return
    return sentence_embeddings.cpu().numpy()

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
    """Print all available recommended Hugging Face embedding models with descriptions"""
    print("\nAvailable Hugging Face embedding models:")
    print("-" * 80)
    for model, desc in HF_EMBEDDING_MODELS.items():
        print(f"{model}")
        print(f"    {desc}")
    print("-" * 80)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Hugging Face Embedding Similarity Demo")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                        help="Hugging Face model to use for embeddings")
    parser.add_argument("--list-models", action="store_true",
                        help="List all available recommended models and exit")
    parser.add_argument("--custom-model", type=str, 
                        help="Use a custom model from Hugging Face Hub")
    args = parser.parse_args()
    
    # If --list-models flag is provided, list models and exit
    if args.list_models:
        list_models()
        return
    
    # Determine which model to use
    model_name = args.custom_model if args.custom_model else args.model
    
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
    print("Generating Hugging Face embeddings...")
    all_sentences = similar_pair + dissimilar_pair + mixed_examples
    
    try:
        all_embeddings = get_huggingface_embeddings(all_sentences, model_name=model_name)
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
    print("\nTip: Run with --list-models to see all available recommended Hugging Face models")
    print("Example: python huggingface_embeddings_demo.py --model roberta-base")
    print("Example with custom model: python huggingface_embeddings_demo.py --custom-model intfloat/multilingual-e5-large")