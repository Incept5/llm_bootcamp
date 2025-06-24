
# LLM Workshop: Comprehensive Overview and Deep Dive

This repository contains code samples for a hands-on workshop that provides a comprehensive overview of Large Language Model (LLM) concepts and demonstrates how LLMs work under the hood.

## Workshop Structure

The workshop is organized into two main sections:

### 1. **Code Samples** (`code/` directory)
Production-ready examples showing different ways to interact with LLMs

### 2. **Interactive Demos** (`demos/` directory)
Educational tools that demonstrate core LLM concepts through hands-on experimentation

---

## Prerequisites

### System Requirements
- Python 3.8 or higher
- At least 8GB RAM (16GB recommended for local models)
- Internet connection for API-based examples

### Required Services

#### Ollama (for local LLM inference)
1. Install Ollama from [https://ollama.ai](https://ollama.ai)
2. Start Ollama service:
   ```bash
   ollama serve
   ```
3. Pull required models:
   ```bash
   ollama pull qwen3:0.6b
   ollama pull qwen3:4b
   ollama pull all-minilm:33m-l12-v2-fp16
   ```

#### Groq API (for cloud-based inference)
1. Sign up at [https://console.groq.com](https://console.groq.com)
2. Get your API key
3. Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

#### Kaggle (for dataset access)
1. Create a Kaggle account at [https://www.kaggle.com](https://www.kaggle.com)
2. For API access, go to Account > API > Create New Token
3. Place the downloaded `kaggle.json` in your home directory under `.kaggle/`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json` (Linux/macOS)

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd munich_ai_2025
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the system dictionary (for word embeddings demo):
   - **macOS/Linux**: Usually available at `/usr/share/dict/words`
   - **Windows**: You may need to download a word list file

---

## Code Samples (`code/` directory)

### 1. **Groq Cloud API Examples**

#### `groq_astrologer.py` - AI Astrology Assistant
Demonstrates structured prompting and rich text output formatting.

```bash
python code/groq_astrologer.py
```

**What it teaches:**
- System prompts vs user prompts
- Structured output formatting (Markdown)
- Rich console output with the `rich` library
- API key management with environment variables

#### `groq_llama.py` - Basic Groq Integration
Simple example of Groq API usage.

```bash
python code/groq_llama.py
```

### 2. **Local LLM Examples**

#### `LLM_Locally.py` - Ollama Performance Testing
Benchmarks local LLM performance across different types of tasks.

```bash
python code/LLM_Locally.py
```

**What it teaches:**
- Local vs cloud LLM trade-offs
- Performance measurement and timing
- Different types of reasoning tasks
- Model initialization overhead

#### `ollama_astrology.py` - Local AI Assistant with Reasoning
Advanced example showing the "thinking" feature in local models.

```bash
python code/ollama_astrology.py
```

**What it teaches:**
- Chain-of-thought reasoning
- Model thinking process visualization
- Advanced Ollama features
- Temperature and sampling parameters

#### `qwen3.py` - Minimal Ollama Example
Simplest possible Ollama integration.

```bash
python code/qwen3.py
```

### 3. **Data Processing Examples**

#### `load_kaggle.py` - Kaggle Dataset Loader
Utility for downloading and loading Kaggle datasets for analysis.

```bash
python code/load_kaggle.py
```

**What it teaches:**
- Kaggle dataset integration with `kagglehub`
- Automatic CSV file discovery and loading
- Pandas DataFrame manipulation
- Dataset exploration and inspection
- File system navigation and data handling

**Features:**
- Downloads datasets from Kaggle using the `kagglehub` library
- Automatically finds and loads CSV files from downloaded datasets
- Provides dataset shape and preview functionality
- Includes JSON export capabilities for data inspection
- Example implementation using Trump Tweets dataset

**Prerequisites:**
- Kaggle account (for dataset access)
- Internet connection for dataset downloads

---

## Interactive Demos (`demos/` directory)

### 1. **Tokenization and Token Probabilities**

#### `token_test.py` - Understanding Tokenization
Demonstrates how text is broken down into tokens.

```bash
python demos/token_test.py
```

**What it teaches:**
- How LLMs process text
- Token IDs and their meaning
- Tokenization vs detokenization
- Subword tokenization concepts

#### `token_probs.py` - Next Token Prediction Interface
Interactive web interface for exploring token probability distributions.

```bash
python demos/token_probs.py
```

**What it teaches:**
- How LLMs generate text token by token
- Probability distributions over vocabulary
- Temperature, top-k, and top-p sampling
- Interactive model exploration

### 2. **Embeddings and Semantic Similarity**

#### `embedding_example.py` - Sentence Similarity Matching
Find semantically similar sentences using embeddings.

```bash
python demos/embedding_example.py
```

**What it teaches:**
- Vector embeddings concept
- Cosine similarity calculation
- Semantic search fundamentals
- Local embedding generation

#### `word_embeddings.py` - 3D Word Embedding Visualization
Generate and visualize word embeddings in 3D space.

```bash
python demos/word_embeddings.py
```

**What it teaches:**
- High-dimensional vector spaces
- Principal Component Analysis (PCA)
- Semantic relationships in vector space
- Interactive 3D visualization

**Note:** This creates a `3d_plot.html` file that opens in your browser.

#### `embedding_demo2.py` - Advanced Embedding Operations
Extended embedding examples with more sophisticated operations.

```bash
python demos/embedding_demo2.py
```

#### `qwen3-embedding-demo.py` - RAG Search with Grimm Fairy Tales
Demonstrates Retrieval-Augmented Generation (RAG) using Qwen3 embeddings for semantic search.

```bash
python demos/qwen3-embedding-demo.py
```

**Prerequisites:**
- Requires `Kinder- und Hausmärchen der Gebrüder Grimm.txt` file in the project root
- Needs Ollama with `qwen3` model installed

**What it teaches:**
- RAG (Retrieval-Augmented Generation) concepts
- Document chunking and preprocessing
- Semantic search with embeddings
- Context-aware answer generation
- Token counting and management
- Batch processing for efficiency

### 3. **Traditional Machine Learning**

#### `ml-demo.py` - Neural Network from Scratch
Classic MNIST digit classification to contrast with modern LLMs.

```bash
python demos/ml-demo.py
```

**What it teaches:**
- Traditional neural networks
- Supervised learning concepts
- How LLMs differ from traditional ML
- TensorFlow/Keras basics

#### `test_keras_fix.py` - TensorFlow Compatibility
Ensures TensorFlow is properly configured.

```bash
python demos/test_keras_fix.py
```

---

## Workshop Learning Path

### Beginner Track
1. Start with `code/qwen3.py` - simplest LLM interaction
2. Try `demos/token_test.py` - understand tokenization
3. Run `code/groq_astrologer.py` - see structured prompting
4. Explore `demos/embedding_example.py` - learn about embeddings
5. Try `code/load_kaggle.py` - learn data loading and processing

### Intermediate Track
1. Run `code/LLM_Locally.py` - compare local vs cloud performance
2. Use `demos/token_probs.py` - deep dive into text generation
3. Try `demos/word_embeddings.py` - visualize semantic relationships
4. Experiment with `code/ollama_astrology.py` - advanced local features

### Advanced Track
1. Run `demos/qwen3-embedding-demo.py` - explore RAG systems
2. Modify sampling parameters in `token_probs.py`
3. Create custom embedding visualizations
4. Compare different model architectures
5. Implement your own LLM applications

---

## Troubleshooting

### Common Issues

#### "Model not found" errors
- Ensure Ollama is running: `ollama serve`
- Pull required models: `ollama pull qwen3:0.6b`

#### Groq API errors
- Check your API key in `.env` file
- Verify internet connection
- Check Groq service status

#### Memory issues
- Use smaller models (qwen3:0.6b instead of larger variants)
- Close other applications
- Reduce batch sizes in embedding operations

#### Missing dictionary file
- **macOS/Linux**: Install `wamerican` or similar package
- **Windows**: Download a word list and update the path in `word_embeddings.py`

#### Missing Grimm fairy tales text file
- The `qwen3-embedding-demo.py` requires `Kinder- und Hausmärchen der Gebrüder Grimm.txt`
- Download from Project Gutenberg or other public domain sources
- Place in the project root directory

### Performance Tips

- **Local models**: Use GPU acceleration if available
- **Embeddings**: Results are cached in `embeddings.pkl` to avoid recomputation
- **Token probabilities**: Start with smaller top-k values for faster computation

---

## Key Concepts Covered

### 1. **LLM Fundamentals**
- Tokenization and vocabulary
- Transformer architecture basics
- Attention mechanisms
- Token generation process

### 2. **Embeddings and Similarity**
- Vector representations of text
- Semantic similarity calculations
- Dimensionality reduction techniques
- Practical search applications

### 3. **LLM Interaction Patterns**
- System prompts vs user prompts
- Temperature and sampling strategies
- Structured output generation
- Chain-of-thought reasoning

### 4. **Deployment Options**
- Local inference with Ollama
- Cloud APIs (Groq)
- Performance trade-offs
- Cost considerations

### 5. **Advanced Techniques**
- Retrieval-Augmented Generation (RAG)
- Document chunking and preprocessing
- Context-aware answer generation
- Semantic search applications

### 6. **Data Integration**
- Kaggle dataset integration
- Automated data discovery and loading
- CSV processing and DataFrame operations
- Data exploration and visualization preparation

---

## Next Steps

After completing the workshop:

1. **Experiment with different models** - Try larger models when available
2. **Build your own applications** - Use the patterns shown in the examples
3. **Explore advanced techniques** - RAG, fine-tuning, agent frameworks
4. **Join the community** - Participate in LLM research and development

---

## Resources

- [Ollama Documentation](https://ollama.ai/docs)
- [Groq API Documentation](https://console.groq.com/docs)
- [Transformers Library](https://huggingface.co/transformers/)
- [Sentence Transformers](https://www.sbert.net/)

---

## Contributing

Feel free to submit improvements, additional examples, or bug fixes via pull requests.

## License

This educational content is provided for learning purposes. Please respect the licenses of the underlying models and libraries used.
