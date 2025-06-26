
# Introduction to LLMs - Configuration Test Scripts

This directory contains simple Python scripts designed to verify that your environment is properly configured to interact with Large Language Models (LLMs) through Ollama.

## Purpose

These scripts serve as initial configuration tests to ensure:
- Ollama is installed and running locally
- Required models are downloaded and available
- Python environment can successfully communicate with the Ollama API
- Basic LLM functionality is working as expected

## Prerequisites

Before running these scripts, ensure you have:

1. **Ollama installed** - Download from [ollama.ai](https://ollama.ai)
2. **Ollama running** - Start with `ollama serve` in your terminal
3. **Required models downloaded** - Pull models using `ollama pull <model-name>`
4. **Python environment** - With required dependencies installed

## Scripts

### `qwen3.py`
A basic test script that:
- Connects to the local Ollama API (http://localhost:11434)
- Uses the Qwen3 1.7B model
- Sends a simple "Hello" prompt
- Displays the response or any error messages

**Usage:**
```bash
python qwen3.py
```

**Expected Output:**
If everything is configured correctly, you should see a greeting response from the Qwen3 model.

**Troubleshooting:**
- If you get a connection error, ensure Ollama is running (`ollama serve`)
- If you get a model not found error, download the model (`ollama pull qwen3:1.7b`)
- Check that the Ollama API is accessible at `http://localhost:11434`

## Dependencies

The scripts in this directory require:
- `requests` library for HTTP communication with Ollama

Install with:
```bash
pip install requests
```

## What's Next

This directory will be expanded with additional test scripts covering:
- Different model types and sizes
- Various prompt formats
- Streaming responses
- Error handling patterns
- Performance benchmarks

## Quick Start

1. Ensure Ollama is running:
   ```bash
   ollama serve
   ```

2. Download the Qwen3 model:
   ```bash
   ollama pull qwen3:1.7b
   ```

3. Run the test script:
   ```bash
   cd intro_to_llms
   python qwen3.py
   ```

If you see a response from the model, your environment is ready for more advanced LLM development!
