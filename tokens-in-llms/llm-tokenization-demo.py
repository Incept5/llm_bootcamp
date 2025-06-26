import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer, GPT2Tokenizer, T5Tokenizer, BertTokenizer


def demonstrate_tokenization():
    """
    A function to demonstrate how tokenization works in LLMs with
    visualizations and examples using different tokenizers (GPT2, T5, BERT)
    without using the tiktoken package.
    """
    print("Demonstrating Tokenization in Large Language Models")
    print("=" * 50)

    # Example text to tokenize
    example_texts = [
        "Hello, world!",
        "Neural networks are transforming AI.",
        "Large language models use subword tokenization.",
        "こんにちは世界",  # Hello world in Japanese
        "LLMs like Claude, GPT-4, and BERT all use different tokenization approaches."
    ]

    # ==================== GPT2-style Tokenization ====================
    print("\n1. GPT2-Style Tokenization")
    print("-" * 50)

    # Initialize GPT2 tokenizer
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Show tokenization results for our example texts
    for i, text in enumerate(example_texts):
        tokens = gpt2_tokenizer.encode(text)
        decoded = [gpt2_tokenizer.decode([token]) for token in tokens]

        print(f"\nExample {i + 1}: '{text}'")
        print(f"Token IDs: {tokens}")
        print(f"Token count: {len(tokens)}")
        print(f"Decoded tokens: {decoded}")

    # ==================== T5-style Tokenization ====================
    # print("\n\n2. T5-Style Tokenization")
    # print("-" * 50)
    #
    # # Initialize T5 tokenizer with explicit model_max_length and legacy=False
    # t5_tokenizer = T5Tokenizer.from_pretrained("t5-base", model_max_length=1024, legacy=False)
    #
    # for i, text in enumerate(example_texts):
    #     encoded = t5_tokenizer.encode(text)
    #     tokens = t5_tokenizer.convert_ids_to_tokens(encoded)
    #
    #     print(f"\nExample {i + 1}: '{text}'")
    #     print(f"Token IDs: {encoded}")
    #     print(f"Token count: {len(encoded)}")
    #     print(f"Decoded tokens: {tokens}")

    # ==================== BERT-style Tokenization ====================
    print("\n\n3. BERT-Style Tokenization")
    print("-" * 50)

    # Initialize BERT tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    for i, text in enumerate(example_texts):
        encoded = bert_tokenizer.encode(text)
        tokens = bert_tokenizer.convert_ids_to_tokens(encoded)

        print(f"\nExample {i + 1}: '{text}'")
        print(f"Token IDs: {encoded}")
        print(f"Token count: {len(encoded)}")
        print(f"Decoded tokens: {tokens}")

    # ==================== Visual Comparison ====================
    print("\n\n4. Visual Comparison of Tokenizers")
    print("-" * 50)

    # Prepare data for visualization
    token_counts = {
        'Text': [f"Example {i + 1}" for i in range(len(example_texts))],
        'GPT2': [len(gpt2_tokenizer.encode(text)) for text in example_texts],
        # 'T5': [len(t5_tokenizer.encode(text)) for text in example_texts],
        'BERT': [len(bert_tokenizer.encode(text)) for text in example_texts]
    }

    token_df = pd.DataFrame(token_counts)

    # Create a bar chart comparing token counts
    plt.figure(figsize=(12, 6))
    x = np.arange(len(token_df['Text']))
    width = 0.25

    plt.bar(x - width, token_df['GPT2'], width, label='GPT2')
    # plt.bar(x, token_df['T5'], width, label='T5')
    plt.bar(x + width, token_df['BERT'], width, label='BERT')

    plt.xlabel('Examples')
    plt.ylabel('Token Count')
    plt.title('Token Count Comparison: GPT2 vs T5 vs BERT')
    plt.xticks(x, token_df['Text'], rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.savefig('tokenization_comparison.png')
    print("Visualization saved as 'tokenization_comparison.png'")

    # ==================== Special Cases ====================
    print("\n\n5. Special Cases and Edge Examples")
    print("-" * 50)

    special_cases = [
        "1234567890",  # Numbers
        "https://www.example.com",  # URLs
        "She said, \"Hello!\"",  # Quotes
        "Python is easy-to-learn",  # Hyphens
        "🙂👍🔥"  # Emojis
    ]

    print("\nGPT2 Tokenization of Special Cases:")
    for case in special_cases:
        gpt2_tokens = gpt2_tokenizer.encode(case)
        print(f"'{case}' → {len(gpt2_tokens)} tokens: {[gpt2_tokenizer.decode([t]) for t in gpt2_tokens]}")

    # print("\nT5 Tokenization of Special Cases:")
    # for case in special_cases:
    #     t5_encoded = t5_tokenizer.encode(case)
    #     t5_tokens = t5_tokenizer.convert_ids_to_tokens(t5_encoded)
    #     print(f"'{case}' → {len(t5_encoded)} tokens: {t5_tokens}")

    print("\nBERT Tokenization of Special Cases:")
    for case in special_cases:
        bert_encoded = bert_tokenizer.encode(case)
        bert_tokens = bert_tokenizer.convert_ids_to_tokens(bert_encoded)
        print(f"'{case}' → {len(bert_encoded)} tokens: {bert_tokens}")

    # ==================== Tokenization Algorithms ====================
    print("\n\n6. Tokenization Algorithm Comparison")
    print("-" * 50)
    print("GPT2: Uses Byte-Pair Encoding (BPE)")
    print("T5: Uses SentencePiece with unigram language model")
    print("BERT: Uses WordPiece tokenization")

    # Usage example
    example = "Tokenization is fundamental to how language models process text."
    print(f"\nExample: '{example}'")

    print("\nGPT2 (BPE):")
    gpt2_tokens = gpt2_tokenizer.tokenize(example)
    print(f"Tokens: {gpt2_tokens}")

    # print("\nT5 (SentencePiece):")
    # t5_tokens = t5_tokenizer.tokenize(example)
    # print(f"Tokens: {t5_tokens}")

    print("\nBERT (WordPiece):")
    bert_tokens = bert_tokenizer.tokenize(example)
    print(f"Tokens: {bert_tokens}")


if __name__ == "__main__":
    demonstrate_tokenization()