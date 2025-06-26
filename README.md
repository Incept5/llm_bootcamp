
# LLM Bootcamp: Hands-on GenAI Development Workshop

This repository contains a comprehensive collection of code samples, interactive demos, and educational materials for understanding and working with Large Language Models (LLMs). The workshop takes you from basic LLM interactions to advanced techniques like Retrieval-Augmented Generation (RAG) and data extraction.

## Learning Path

Follow these directories in order for the optimal learning experience. Each directory focuses on specific concepts and builds upon previous knowledge:

1. **[intro-to-llms/](intro-to-llms/)** - Start here: Basic LLM interactions, cloud vs local models
2. **[ui-for-local-llms/](ui-for-local-llms/)** - Building user interfaces for LLM applications  
3. **[machine-learning/](machine-learning/)** - Traditional ML concepts and TensorFlow integration
4. **[tokens-in-llms/](tokens-in-llms/)** - Understanding tokenization and text generation mechanics
5. **[embedding/](embedding/)** - Vector embeddings, semantic similarity, and 3D visualizations
6. **[reasoning/](reasoning/)** - Chain-of-thought reasoning and AI decision-making processes
7. **[retrieval-augmented-generation/](retrieval-augmented-generation/)** - RAG systems with ChromaDB and semantic search
8. **[extracting-data/](extracting-data/)** - Structured data extraction and web scraping with AI
9. **[mcp/](mcp/)** - Model Context Protocol for advanced integrations
10. **[extras/](extras/)** - Additional examples and advanced techniques

## Directory Overview

### Core Learning Modules

- **intro-to-llms/** - Foundation concepts: local vs cloud LLMs, basic interactions, performance testing
- **ui-for-local-llms/** - Building web interfaces with Gradio for LLM applications
- **machine-learning/** - Traditional neural networks and TensorFlow integration to understand LLM foundations
- **tokens-in-llms/** - Deep dive into tokenization, next token prediction, and text generation mechanics
- **embedding/** - Vector embeddings, semantic similarity, word relationships, and 3D visualizations
- **reasoning/** - Chain-of-thought reasoning, AI astrology examples, and complex decision processes
- **retrieval-augmented-generation/** - Complete RAG systems using ChromaDB, document chunking, and semantic search
- **extracting-data/** - AI-powered data extraction, web scraping, and structured output generation
- **mcp/** - Model Context Protocol demonstrations and advanced integrations
- **extras/** - Additional examples including fill-in-middle, formatted output, and advanced techniques

### Supporting Files

- **Text Files** - Sample documents for RAG demonstrations (Alice in Wonderland, Grimm Fairy Tales)
- **Configuration** - Environment setup, requirements, and model configurations
- **Documentation** - Individual README files in each directory provide detailed explanations

## Quick Start

### Prerequisites
- Python 3.8+ with pip
- 8GB+ RAM (16GB recommended for local models)
- Internet connection for cloud APIs and model downloads

### Installation

1. **Clone and install dependencies:**
   ```bash
   git clone [repository-url]
   cd llm_bootcamp
   pip install -r requirements.txt
   ```

2. **Set up Ollama (for local models):**
   - Install from [ollama.ai](https://ollama.ai)
   - Start service: `ollama serve`
   - Pull basic models:
     ```bash
     ollama pull qwen3:0.6b
     ollama pull nomic-embed-text
     ```

3. **Optional: Set up cloud APIs:**
   - **Groq**: Get API key from [console.groq.com](https://console.groq.com), add to `.env` file
   - **Kaggle**: Get API credentials from [kaggle.com](https://www.kaggle.com), place `kaggle.json` in `~/.kaggle/`

### Getting Started

Start with the first directory and follow the learning path:
```bash
cd intro-to-llms
python local_llm_using_ollama.py
```

Each directory contains its own README with detailed setup instructions and explanations.

## What You'll Learn

This bootcamp covers the full spectrum of LLM development:

### Core Concepts
- **LLM Fundamentals**: Tokenization, transformer architecture, text generation mechanics
- **Embeddings**: Vector representations, semantic similarity, 3D visualizations
- **Local vs Cloud**: Trade-offs between local models (Ollama) and cloud APIs (Groq)
- **User Interfaces**: Building web interfaces with Gradio for LLM applications

### Advanced Techniques
- **Reasoning**: Chain-of-thought prompting, complex decision-making processes
- **RAG Systems**: Document chunking, semantic search, context-aware generation
- **Data Extraction**: Structured output generation, web scraping with AI
- **Integrations**: Model Context Protocol (MCP) and advanced tool usage

### Practical Skills
- Performance optimization and model comparison
- Traditional ML foundations with TensorFlow
- Real-world data processing with Kaggle datasets
- Production-ready RAG implementations with ChromaDB
- AI-powered SQL query generation

## Troubleshooting

### Common Issues
- **Model not found**: Ensure Ollama is running (`ollama serve`) and models are pulled
- **API errors**: Check API keys in `.env` file and internet connection
- **Memory issues**: Use smaller models (qwen3:0.6b) and reduce batch sizes
- **Missing files**: Download required text files (Alice in Wonderland, Grimm Fairy Tales) as noted in individual READMEs

### Performance Tips
- Use GPU acceleration if available for local models
- Embedding results are cached to avoid recomputation
- Start with smaller models and parameters for faster experimentation

## Resources

- [Ollama Documentation](https://ollama.ai/docs) - Local LLM setup and management
- [Groq API Documentation](https://console.groq.com/docs) - Cloud LLM service
- [Hugging Face Transformers](https://huggingface.co/transformers/) - Model library and tools
- [Sentence Transformers](https://www.sbert.net/) - Embedding models and techniques

## Contributing

Contributions are welcome! Please submit improvements, additional examples, or bug fixes via pull requests.

---

*This educational content is provided for learning purposes. Please respect the licenses of the underlying models and libraries used.*
