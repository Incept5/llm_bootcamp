"""
Demonstration of tokenization using tiktoken, the tokenizer used by OpenAI models.
This script shows how to use tiktoken to tokenize text for GPT models and compare
different OpenAI tokenizers.
"""

import tiktoken
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import time

def list_available_encodings():
    """List all available tiktoken encodings"""
    print("Available tiktoken encodings:")
    for encoding_name in tiktoken.list_encoding_names():
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            print(f"  - {encoding_name}")
        except Exception as e:
            print(f"  - {encoding_name} (error: {e})")
    
    print("\nCommon model-to-encoding mappings:")
    print("  - gpt-4, gpt-3.5-turbo --> cl100k_base")
    print("  - text-embedding-ada-002 --> cl100k_base")
    print("  - text-davinci-003, text-davinci-002 --> p50k_base")
    print("  - davinci, curie, babbage, ada --> r50k_base (or p50k_base for older models)")
    print("  - gpt-2 --> gpt2")

def demonstrate_tiktoken(encoding_name, texts):
    """Demonstrate tokenization using the specified tiktoken encoding"""
    try:
        # Get the encoding
        encoding = tiktoken.get_encoding(encoding_name)
        print(f"\nUsing encoding: {encoding_name}")
        
        # Process each text
        for i, text in enumerate(texts):
            print(f"\nText {i+1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            # Time the tokenization
            start_time = time.time()
            tokens = encoding.encode(text)
            elapsed_time = (time.time() - start_time) * 1000  # ms
            
            # Print results
            print(f"Token count: {len(tokens)}")
            print(f"Token IDs: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
            
            # Decode tokens to show how the text was split
            decoded_tokens = []
            for token in tokens[:20]:  # Limit to first 20 tokens for display
                decoded = encoding.decode([token])
                # Replace whitespace with visible markers for clarity
                decoded = decoded.replace(' ', '·')
                decoded = decoded.replace('\n', '\\n')
                decoded_tokens.append(decoded)
            
            print(f"Decoded tokens: {decoded_tokens}{'...' if len(tokens) > 20 else ''}")
            print(f"Tokenization time: {elapsed_time:.2f} ms")
            
        return True
    except Exception as e:
        print(f"Error with encoding {encoding_name}: {str(e)}")
        return False

def compare_encodings(texts):
    """Compare different tiktoken encodings on the same texts"""
    # Common encodings to compare
    encodings = ["cl100k_base", "p50k_base", "r50k_base", "gpt2"]
    
    results = []
    
    # Process each text with each encoding
    for text_idx, text in enumerate(texts):
        for enc_name in encodings:
            try:
                encoding = tiktoken.get_encoding(enc_name)
                
                # Time the tokenization
                start_time = time.time()
                tokens = encoding.encode(text)
                elapsed_time = (time.time() - start_time) * 1000  # ms
                
                # Store results
                results.append({
                    "encoding": enc_name,
                    "text_idx": text_idx,
                    "token_count": len(tokens),
                    "time_ms": elapsed_time
                })
                
            except Exception as e:
                print(f"Error with encoding {enc_name}: {str(e)}")
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Group by encoding and text_idx
    pivot_df = pd.pivot_table(
        df, 
        values="token_count", 
        index="encoding", 
        columns="text_idx",
        aggfunc="first"
    )
    
    # Rename columns to show text snippets
    text_labels = {i: f"Text {i+1}" for i in range(len(texts))}
    pivot_df.columns = [text_labels[i] for i in pivot_df.columns]
    
    # Plot
    pivot_df.plot(kind="bar", ax=plt.gca())
    plt.title("Token Count Comparison Across OpenAI Encodings")
    plt.xlabel("Encoding")
    plt.ylabel("Number of Tokens")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("tiktoken_comparison.png")
    print("\nVisualization saved as 'tiktoken_comparison.png'")
    
    # Print summary table
    print("\nToken count summary:")
    print(tabulate(pivot_df, headers="keys", tablefmt="grid"))
    
    return df

def count_tokens_for_model(model_name, text):
    """Count tokens for a specific OpenAI model"""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        tokens = encoding.encode(text)
        return len(tokens)
    except KeyError:
        print(f"Model {model_name} not found. Using cl100k_base encoding instead.")
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return len(tokens)

def main():
    parser = argparse.ArgumentParser(description="Demonstrate tokenization with tiktoken")
    parser.add_argument("--list", action="store_true", help="List available encodings")
    parser.add_argument("--encoding", type=str, default="cl100k_base", 
                        help="Encoding to use (default: cl100k_base)")
    parser.add_argument("--compare", action="store_true", help="Compare different encodings")
    parser.add_argument("--model", type=str, help="Count tokens for specific OpenAI model")
    parser.add_argument("--text", type=str, help="Text to tokenize")
    parser.add_argument("--file", type=str, help="File containing text samples (one per line)")
    
    args = parser.parse_args()
    
    # Handle listing encodings
    if args.list:
        list_available_encodings()
        return
    
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
    
    # Handle specific model token counting
    if args.model:
        for i, text in enumerate(texts):
            token_count = count_tokens_for_model(args.model, text)
            print(f"\nText {i+1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            print(f"Token count for model {args.model}: {token_count}")
        return
    
    # Handle comparison
    if args.compare:
        compare_encodings(texts)
        return
    
    # Default: demonstrate with specified encoding
    demonstrate_tiktoken(args.encoding, texts)

if __name__ == "__main__":
    try:
        from tabulate import tabulate
    except ImportError:
        def tabulate(df, **kwargs):
            return str(df)
        print("Note: Install 'tabulate' package for better table formatting")
    
    main()