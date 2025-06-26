"""
Demonstration of tokenization using SentencePiece directly.
This script shows how to use SentencePiece for tokenization, which is used by
many models like T5, XLM-R, and others.
"""

import sentencepiece as spm
import argparse
import os
import tempfile
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def train_simple_model(texts, vocab_size=100, model_type="unigram", model_prefix="sp_model"):
    """Train a simple SentencePiece model on the provided texts"""
    # Create a temporary file with the training data
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        for text in texts:
            f.write(text + "\n")
        temp_file = f.name
    
    # Train the model
    print(f"Training SentencePiece model with {model_type} algorithm and vocab size {vocab_size}...")
    spm.SentencePieceTrainer.train(
        f"--input={temp_file} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--model_type={model_type} "
        "--pad_id=0 "
        "--unk_id=1 "
        "--bos_id=2 "
        "--eos_id=3 "
        "--character_coverage=1.0"
    )
    
    # Clean up
    os.unlink(temp_file)
    
    print(f"Model trained and saved as {model_prefix}.model and {model_prefix}.vocab")
    return f"{model_prefix}.model"

def demonstrate_sentencepiece(model_path, texts):
    """Demonstrate tokenization using a SentencePiece model"""
    # Load the model
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    print(f"\nLoaded SentencePiece model from {model_path}")
    print(f"Vocabulary size: {sp.get_piece_size()}")
    
    # Show some vocabulary items
    print("\nSample vocabulary items:")
    for i in range(min(20, sp.get_piece_size())):
        print(f"  {i}: '{sp.id_to_piece(i)}' (score: {sp.get_score(i):.4f})")
    
    # Process each text
    for i, text in enumerate(texts):
        print(f"\nText {i+1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Time the tokenization
        start_time = time.time()
        tokens = sp.encode_as_pieces(text)
        token_ids = sp.encode_as_ids(text)
        elapsed_time = (time.time() - start_time) * 1000  # ms
        
        # Print results
        print(f"Token count: {len(tokens)}")
        print(f"Tokens: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        print(f"Token IDs: {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}")
        print(f"Tokenization time: {elapsed_time:.2f} ms")
        
        # Decode back to text
        decoded = sp.decode_pieces(tokens)
        print(f"Decoded text: '{decoded[:50]}{'...' if len(decoded) > 50 else ''}'")
        
        # Check if decoding is perfect
        if decoded == text:
            print("✓ Perfect reconstruction")
        else:
            print("✗ Reconstruction differs from original")

def compare_model_types(texts, vocab_sizes=[1000, 5000]):
    """Compare different SentencePiece model types and vocabulary sizes"""
    model_types = ["unigram", "bpe", "char", "word"]
    results = []
    
    # Train models and collect results
    for model_type in model_types:
        for vocab_size in vocab_sizes:
            try:
                # Skip word model with large vocab as it might not work well
                if model_type == "word" and vocab_size > 1000:
                    continue
                
                model_prefix = f"sp_{model_type}_{vocab_size}"
                
                # Train model
                model_path = train_simple_model(
                    texts, 
                    vocab_size=vocab_size, 
                    model_type=model_type, 
                    model_prefix=model_prefix
                )
                
                # Load model
                sp = spm.SentencePieceProcessor()
                sp.load(model_path)
                
                # Process each text
                for text_idx, text in enumerate(texts):
                    start_time = time.time()
                    tokens = sp.encode_as_pieces(text)
                    elapsed_time = (time.time() - start_time) * 1000  # ms
                    
                    # Store results
                    results.append({
                        "model_type": model_type,
                        "vocab_size": vocab_size,
                        "text_idx": text_idx,
                        "token_count": len(tokens),
                        "time_ms": elapsed_time
                    })
                    
                    # Print brief result
                    print(f"{model_type} (vocab={vocab_size}) - Text {text_idx+1}: {len(tokens)} tokens")
                
            except Exception as e:
                print(f"Error with {model_type} model (vocab={vocab_size}): {str(e)}")
    
    # Create visualizations
    if results:
        df = pd.DataFrame(results)
        
        # Plot token counts by model type and vocab size
        plt.figure(figsize=(12, 8))
        
        # Group by model_type, vocab_size, and text_idx
        pivot_df = pd.pivot_table(
            df, 
            values="token_count", 
            index=["model_type", "vocab_size"], 
            columns="text_idx",
            aggfunc="first"
        )
        
        # Rename columns
        pivot_df.columns = [f"Text {i+1}" for i in pivot_df.columns]
        
        # Plot
        pivot_df.plot(kind="bar", ax=plt.gca())
        plt.title("Token Count Comparison Across SentencePiece Models")
        plt.xlabel("Model Type and Vocabulary Size")
        plt.ylabel("Number of Tokens")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("sentencepiece_comparison.png")
        print("\nVisualization saved as 'sentencepiece_comparison.png'")
        
        # Clean up model files
        for model_type in model_types:
            for vocab_size in vocab_sizes:
                for ext in [".model", ".vocab"]:
                    try:
                        os.remove(f"sp_{model_type}_{vocab_size}{ext}")
                    except:
                        pass
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Demonstrate SentencePiece tokenization")
    parser.add_argument("--train", action="store_true", help="Train a new SentencePiece model")
    parser.add_argument("--model-type", type=str, default="unigram", 
                        choices=["unigram", "bpe", "char", "word"],
                        help="Model type for training (default: unigram)")
    parser.add_argument("--vocab-size", type=int, default=1000, 
                        help="Vocabulary size for training (default: 1000)")
    parser.add_argument("--model", type=str, help="Path to existing SentencePiece model")
    parser.add_argument("--compare", action="store_true", 
                        help="Compare different model types and vocabulary sizes")
    parser.add_argument("--text", type=str, help="Text to tokenize")
    parser.add_argument("--file", type=str, help="File containing text samples (one per line)")
    
    args = parser.parse_args()
    
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
    
    # Handle comparison
    if args.compare:
        compare_model_types(texts)
        return
    
    # Handle training or using existing model
    model_path = None
    
    if args.train:
        model_prefix = "sp_model"
        model_path = train_simple_model(
            texts, 
            vocab_size=args.vocab_size, 
            model_type=args.model_type,
            model_prefix=model_prefix
        )
    elif args.model:
        model_path = args.model
    else:
        # If no model specified, train a default one
        model_prefix = "sp_model"
        model_path = train_simple_model(texts, model_prefix=model_prefix)
    
    # Demonstrate tokenization
    demonstrate_sentencepiece(model_path, texts)
    
    # Clean up model files if we created them
    if args.train or not args.model:
        try:
            os.remove(f"{model_path}")
            os.remove(f"{model_path.replace('.model', '.vocab')}")
        except:
            pass

if __name__ == "__main__":
    main()