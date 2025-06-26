
# Tokenization in Large Language Models

This directory contains demonstrations and examples of tokenization - the fundamental process that converts human-readable text into numerical tokens that language models can understand and process.

## What is Tokenization?

Tokenization is the process of breaking down text into smaller units called "tokens" that can be processed by machine learning models. In the context of Large Language Models (LLMs), tokenization serves as the bridge between human language and the numerical representations that neural networks operate on.

### Why Tokenization Matters

- **Input Processing**: LLMs can only work with numerical data, so text must be converted to numbers
- **Vocabulary Management**: Tokenization helps manage the vocabulary size while preserving meaning
- **Efficiency**: Good tokenization balances between having enough detail and computational efficiency
- **Multilingual Support**: Modern tokenization handles multiple languages and special characters
- **Subword Handling**: Deals with out-of-vocabulary words by breaking them into smaller, known pieces

## Types of Tokenization

### 1. Word-Level Tokenization
- **Method**: Splits text by whitespace and punctuation
- **Pros**: Simple, interpretable
- **Cons**: Large vocabulary, can't handle out-of-vocabulary words
- **Example**: "Hello world!" → ["Hello", "world", "!"]

### 2. Character-Level Tokenization
- **Method**: Each character becomes a token
- **Pros**: Small vocabulary, no out-of-vocabulary issues
- **Cons**: Very long sequences, loses word-level meaning
- **Example**: "Hello" → ["H", "e", "l", "l", "o"]

### 3. Subword Tokenization (Modern Approach)
Most current LLMs use subword tokenization methods:

#### Byte-Pair Encoding (BPE)
- **Used by**: GPT family (GPT-2, GPT-3, GPT-4), CodeLlama, many others
- **Method**: Iteratively merges the most frequent character pairs
- **Pros**: Good balance between vocabulary size and sequence length
- **Cons**: Can split words in unintuitive ways
- **Example**: "tokenization" → ["token", "ization"] or ["tok", "en", "ization"]

#### WordPiece
- **Used by**: BERT family, DistilBERT, RoBERTa
- **Method**: Similar to BPE but uses likelihood-based merging
- **Pros**: Better handles morphologically rich languages
- **Cons**: More complex training process
- **Example**: "tokenization" → ["token", "##ization"] (## indicates continuation)

#### SentencePiece
- **Used by**: T5, XLM-R, mT5, many multilingual models
- **Method**: Treats text as raw bytes, includes multiple algorithms (Unigram, BPE)
- **Pros**: Language-agnostic, handles any Unicode text
- **Cons**: Can be less intuitive for single languages
- **Example**: "tokenization" → ["▁token", "ization"] (▁ represents space)

#### Unigram Language Model
- **Used by**: Some SentencePiece implementations, XLNet
- **Method**: Uses probabilistic approach to find optimal segmentation
- **Pros**: Theoretically optimal segmentation
- **Cons**: More complex, computationally intensive

## Demo Files: From Simple to Advanced

### 🟢 **Simple (Beginner)**

#### 1. `simple_token_test.py`
**Purpose**: Basic introduction to tokenization concepts
- Shows fundamental tokenize/detokenize operations
- Uses Llama-2 tokenizer as example
- Demonstrates token ID to text mapping
- **Run**: `python simple_token_test.py`
- **Learning**: Basic tokenization workflow

#### 2. `simple_tokenizer_comparison.py`
**Purpose**: Compare two different tokenizers side-by-side
- Compares GPT-2 vs OPT tokenizers
- Shows how different models tokenize the same text differently
- **Run**: `python simple_tokenizer_comparison.py`
- **Learning**: Different models = different tokenization

### 🟡 **Intermediate**

#### 3. `tiktoken_demo.py`
**Purpose**: Deep dive into OpenAI's tiktoken library
- Comprehensive demonstration of OpenAI tokenizers (GPT-3.5, GPT-4)
- Performance timing and analysis
- Multiple encoding comparisons (cl100k_base, p50k_base, etc.)
- Visualization of token count differences
- Special case handling (URLs, emojis, multilingual text)
- **Run**: `python tiktoken_demo.py --help` (see all options)
- **Features**:
  ```bash
  python tiktoken_demo.py --compare  # Compare encodings
  python tiktoken_demo.py --model gpt-4 --text "Your text"  # Model-specific counting
  python tiktoken_demo.py --list  # Show available encodings
  ```
- **Learning**: Production tokenization, performance considerations

#### 4. `llm-tokenization-demo.py`
**Purpose**: Visual comparison across model families
- Compares GPT-2 (BPE), BERT (WordPiece) tokenization
- Includes special case analysis (numbers, URLs, emojis, multilingual)
- Generates comparison visualizations
- Shows tokenization algorithm differences in practice
- **Run**: `python llm-tokenization-demo.py`
- **Learning**: How different algorithms handle edge cases

#### 5. `sentencepiece_demo.py`
**Purpose**: Hands-on SentencePiece training and usage
- Train custom SentencePiece models from scratch
- Compare different algorithms (Unigram, BPE, Character, Word)
- Vocabulary size experiments
- Model performance analysis
- **Run**: `python sentencepiece_demo.py --help` (see all options)
- **Features**:
  ```bash
  python sentencepiece_demo.py --train --model-type unigram --vocab-size 1000
  python sentencepiece_demo.py --compare  # Compare all model types
  ```
- **Learning**: How tokenizers are actually created

### 🔴 **Advanced**

#### 6. `tokenization_comparison.py`
**Purpose**: Comprehensive analysis framework
- Compare any HuggingFace tokenizers
- Extensive model family coverage (GPT, BERT, T5, multilingual, code-specific)
- Performance benchmarking
- Advanced visualizations and analysis
- Production-ready comparison pipeline
- **Run**: `python tokenization_comparison.py --help`
- **Features**:
  ```bash
  # Compare specific models
  python tokenization_comparison.py --models gpt2 bert-base-uncased t5-base
  
  # Compare entire model families
  python tokenization_comparison.py --category "GPT Family (BPE)"
  
  # Use custom text file
  python tokenization_comparison.py --file your_texts.txt --models gpt2 bert-base-uncased
  
  # List all available model categories
  python tokenization_comparison.py --list-models
  ```
- **Learning**: Production tokenization analysis, model selection

#### 7. `predict_next_token.py`
**Purpose**: Interactive token prediction with probabilities
- Real-time next-token probability distribution
- Interactive Gradio interface
- Temperature, top-p, top-k parameter exploration
- Visual probability analysis with pie charts
- System prompt and context handling
- **Run**: `python predict_next_token.py`
- **Features**:
  - Adjustable sampling parameters
  - Step-by-step token generation
  - Probability visualization
  - Context-aware predictions
- **Learning**: How LLMs actually generate text, sampling strategies

## Getting Started

### Prerequisites
```bash
pip install transformers torch tiktoken sentencepiece matplotlib pandas gradio tabulate
```

### Recommended Learning Path

1. **Start Simple**: Run `simple_token_test.py` to understand basics
2. **Compare Models**: Use `simple_tokenizer_comparison.py` to see differences
3. **Explore OpenAI**: Try `tiktoken_demo.py` with different options
4. **Visual Learning**: Run `llm-tokenization-demo.py` for comprehensive comparison
5. **Hands-on Training**: Experiment with `sentencepiece_demo.py`
6. **Advanced Analysis**: Use `tokenization_comparison.py` for production scenarios
7. **Interactive Exploration**: Launch `predict_next_token.py` for real-time experiments

### Quick Start Examples

```bash
# Basic tokenization
python simple_token_test.py

# Compare OpenAI encodings
python tiktoken_demo.py --compare

# Visual model comparison
python llm-tokenization-demo.py

# Train your own tokenizer
python sentencepiece_demo.py --train --vocab-size 500

# Compare model families
python tokenization_comparison.py --category "GPT Family (BPE)"

# Interactive token prediction
python predict_next_token.py
```

## Key Concepts Demonstrated

### Token Efficiency
- **Shorter sequences**: Better tokenization creates shorter token sequences
- **Vocabulary trade-offs**: Larger vocabularies vs. longer sequences
- **Domain-specific**: Code tokenizers vs. general text tokenizers

### Cross-Model Compatibility
- **Same text, different tokens**: How models tokenize differently
- **Token counting**: Why GPT-4 and BERT count tokens differently
- **Context limits**: How tokenization affects model context windows

### Special Handling
- **Multilingual text**: How tokenizers handle non-English languages
- **Code and structured data**: Special considerations for programming languages
- **Emojis and Unicode**: Modern tokenization challenges

### Performance Considerations
- **Speed**: Tokenization performance across different libraries
- **Memory**: Vocabulary size impact on model memory usage
- **Quality**: Trade-offs between efficiency and semantic preservation

## Further Reading

- [Hugging Face Tokenizers Documentation](https://huggingface.co/docs/tokenizers/)
- [OpenAI Tiktoken Repository](https://github.com/openai/tiktoken)
- [SentencePiece Paper](https://arxiv.org/abs/1808.06226)
- [Neural Machine Translation of Rare Words with Subword Units (BPE Paper)](https://arxiv.org/abs/1508.07909)
- [Google's Neural Machine Translation System (WordPiece)](https://arxiv.org/abs/1609.08144)

## Contributing

When adding new tokenization examples:
1. Follow the complexity progression (Simple → Intermediate → Advanced)
2. Include clear docstrings and comments
3. Add command-line help and examples
4. Update this README with the new file description
5. Consider adding visualization when helpful

---

*This directory provides a complete learning journey through tokenization in LLMs, from basic concepts to advanced production considerations.*
