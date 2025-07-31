import argparse
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import time
import json
from tabulate import tabulate

def list_recommended_models():
    """List recommended tokenizer models to try"""
    models = {
        "GPT Family (BPE)": [
            "gpt2",
            "EleutherAI/gpt-neo-125M",
            "EleutherAI/gpt-j-6B",
            "Salesforce/codegen-350M-mono",
            "bigcode/santacoder",
        ],
        "BERT Family (WordPiece)": [
            "bert-base-uncased",
            "bert-base-cased",
            "distilbert-base-uncased",
            "roberta-base",
            "albert-base-v2",
        ],
        "T5 Family (SentencePiece)": [
            "t5-small",
            "t5-base",
            "google/flan-t5-base",
            "google/mt5-small",
        ],
        "Multilingual": [
            "xlm-roberta-base",
            "facebook/mbart-large-50",
            "facebook/nllb-200-distilled-600M",
            "facebook/m2m100_418M",
            "bert-base-multilingual-cased",
        ],
        "Code-Specific": [
            "codellama/CodeLlama-7b-hf",
            "Salesforce/codegen-350M-mono",
            "bigcode/santacoder",
            "microsoft/codebert-base",
        ],
        "Domain-Specific": [
            "dmis-lab/biobert-v1.1",
            "allenai/scibert_scivocab_uncased",
            "nlpaueb/legal-bert-base-uncased",
            "microsoft/biogpt",
        ]
    }
    
    print("Recommended Tokenizer Models:")
    for category, model_list in models.items():
        print(f"\n{category}:")
        for model in model_list:
            print(f"  - {model}")
    
    return models

def compare_tokenizers(models, texts, save_results=True, visualize=True):
    """Compare tokenization across different models"""
    results = {}
    tokenizers = {}
    
    # Load tokenizers
    print("\nLoading tokenizers...")
    for model in models:
        try:
            tokenizers[model] = AutoTokenizer.from_pretrained(model)
            print(f"✓ Loaded {model}")
        except Exception as e:
            print(f"✗ Failed to load {model}: {str(e)}")
    
    # Process each text with each tokenizer
    all_results = []
    
    for i, text in enumerate(texts):
        print(f"\nProcessing text {i+1}: '{text[:50]}...' if len(text) > 50 else text")
        text_results = []
        
        for model, tokenizer in tokenizers.items():
            start_time = time.time()
            
            # Tokenize
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.encode(text)
            
            # Calculate time
            elapsed_time = (time.time() - start_time) * 1000  # ms
            
            # Store results
            result = {
                "model": model,
                "text_id": i,
                "token_count": len(tokens),
                "tokens": tokens[:10] + ["..."] if len(tokens) > 10 else tokens,
                "time_ms": elapsed_time
            }
            
            text_results.append(result)
            all_results.append(result)
            
            # Print results
            print(f"\n{model}:")
            print(f"  Token count: {len(tokens)}")
            print(f"  First few tokens: {tokens[:10]}")
            print(f"  Time: {elapsed_time:.2f} ms")
        
        # Sort results by token count for this text
        text_results.sort(key=lambda x: x["token_count"])
        results[f"text_{i}"] = text_results
    
    # Save results to JSON if requested
    if save_results:
        with open("tokenization_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print("\nResults saved to tokenization_results.json")
    
    # Create visualizations if requested
    if visualize:
        create_visualizations(all_results, texts)
    
    return all_results

def create_visualizations(results, texts):
    """Create visualizations from tokenization results"""
    # Prepare data for token count comparison
    df = pd.DataFrame(results)
    
    # Group by text_id and model
    pivot_df = pd.pivot_table(
        df, 
        values="token_count", 
        index="model", 
        columns="text_id",
        aggfunc="first"
    )
    
    # Rename columns to show text snippets
    text_labels = {i: f"Text {i+1}: '{t[:20]}...'" if len(t) > 20 else t for i, t in enumerate(texts)}
    pivot_df.columns = [text_labels[i] for i in pivot_df.columns]
    
    # Plot token counts
    plt.figure(figsize=(12, 8))
    pivot_df.plot(kind="bar", ax=plt.gca())
    plt.title("Token Count Comparison Across Models")
    plt.xlabel("Model")
    plt.ylabel("Number of Tokens")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("token_count_comparison.png")
    print("Visualization saved as 'token_count_comparison.png'")
    
    # Plot tokenization time
    time_df = pd.DataFrame(results)
    time_pivot = pd.pivot_table(
        time_df, 
        values="time_ms", 
        index="model", 
        columns="text_id",
        aggfunc="first"
    )
    
    time_pivot.columns = [text_labels[i] for i in time_pivot.columns]
    
    plt.figure(figsize=(12, 8))
    time_pivot.plot(kind="bar", ax=plt.gca())
    plt.title("Tokenization Time Comparison")
    plt.xlabel("Model")
    plt.ylabel("Time (ms)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("tokenization_time_comparison.png")
    print("Visualization saved as 'tokenization_time_comparison.png'")

def main():
    parser = argparse.ArgumentParser(description="Compare tokenization across different models")
    parser.add_argument("--list-models", action="store_true", help="List recommended models to try")
    parser.add_argument("--models", nargs="+", help="Models to compare (e.g., gpt2 bert-base-uncased t5-base)")
    parser.add_argument("--text", type=str, help="Text to tokenize")
    parser.add_argument("--file", type=str, help="File containing text samples (one per line)")
    parser.add_argument("--category", type=str, help="Use all models from a specific category (e.g., 'GPT Family (BPE)')")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        model_categories = list_recommended_models()
        return
    
    # Determine which models to use
    models_to_use = []
    
    if args.models:
        models_to_use = args.models
    elif args.category:
        model_categories = list_recommended_models()
        if args.category in model_categories:
            models_to_use = model_categories[args.category]
        else:
            print(f"Category '{args.category}' not found. Available categories:")
            for category in model_categories.keys():
                print(f"  - {category}")
            return
    else:
        # Default models if none specified
        models_to_use = ["gpt2", "bert-base-uncased", "t5-base", "xlm-roberta-base"]
    
    # Get text samples
    texts = []
    
    if args.text:
        texts.append(args.text)
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return
    else:
        # Default text samples
        texts = [
            "Tokenization is the first step in processing text for language models.",
            "Different models use different tokenization approaches: BPE, WordPiece, SentencePiece, etc.",
            "Multilingual models need to handle text in many languages: English, 中文, Español, Русский, العربية, etc.",
            "Code tokenization is special: def tokenize(text): return text.split()",
            "URLs and special characters: https://example.com/path?query=value&param=123",
            "Emojis are challenging: 😊 🚀 🌍 🤖 💻"
        ]
    
    # Run comparison
    compare_tokenizers(models_to_use, texts, visualize=not args.no_viz)

if __name__ == "__main__":
    main()