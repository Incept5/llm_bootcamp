
# Introduction to LLMs - Configuration Test Scripts

This directory contains simple Python scripts designed to verify that your environment is properly configured to interact with Large Language Models (LLMs) through both local (Ollama) and cloud-based APIs (Groq).

## Purpose

These scripts serve as initial configuration tests to ensure:
- Local setup: Ollama is installed and running locally with required models
- Cloud setup: API keys are configured for cloud-based LLM services
- Python environment can successfully communicate with both local and cloud APIs
- Basic LLM functionality is working as expected

## Prerequisites

Before running these scripts, ensure you have:

### For Local LLM (Ollama):
1. **Ollama installed** - Download from [ollama.ai](https://ollama.ai)
2. **Ollama running** - Start with `ollama serve` in your terminal
3. **Required models downloaded** - Pull models using `ollama pull <model-name>`

### For Cloud LLM (Groq):
1. **Groq API Key** - Sign up at [groq.com](https://groq.com) and get your API key
2. **Environment variables** - Set up `.env` file with your API key

### General:
3. **Python environment** - With required dependencies installed

## Scripts

### `local_llm_using_ollama.py`
A basic test script that:
- Connects to the local Ollama API (http://localhost:11434)
- Uses the Qwen3 1.7B model
- Sends a simple "Hello" prompt
- Displays the response or any error messages

**Usage:**
```bash
python local_llm_using_ollama.py
```

**Expected Output:**
If everything is configured correctly, you should see a greeting response from the Qwen3 model.

**Troubleshooting:**
- If you get a connection error, ensure Ollama is running (`ollama serve`)
- If you get a model not found error, download the model (`ollama pull qwen3:1.7b`)
- Check that the Ollama API is accessible at `http://localhost:11434`

### `cloud_llm_using_groq.py`
A basic test script that:
- Connects to the Groq cloud API using your API key
- Uses the Llama 3.3 70B Versatile model
- Sends a simple "Hello" prompt
- Displays the response from the cloud-based model

**Usage:**
```bash
python cloud_llm_using_groq.py
```

**Setup Requirements:**
1. Create a `.env` file in the project root with:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
2. Ensure you have a valid Groq API key from [groq.com](https://groq.com)

**Expected Output:**
If everything is configured correctly, you should see a greeting response from the Llama 3.3 70B model.

**Troubleshooting:**
- If you get an authentication error, check your API key in the `.env` file
- Ensure the `.env` file is in the correct location (project root)
- Verify your Groq account has API access and sufficient credits

## Dependencies

The scripts in this directory require:
- `requests` library for HTTP communication with Ollama
- `groq` library for Groq API communication
- `python-dotenv` library for environment variable management

Install with:
```bash
pip install requests groq python-dotenv
```

## What's Next

This directory will be expanded with additional test scripts covering:
- Different model types and sizes (both local and cloud)
- Various prompt formats and conversation patterns
- Streaming responses
- Error handling patterns
- Performance benchmarks comparing local vs cloud models
- Cost analysis for cloud API usage

## Quick Start

### Local LLM Setup (Ollama):
1. Ensure Ollama is running:
   ```bash
   ollama serve
   ```

2. Download the Qwen3 model:
   ```bash
   ollama pull qwen3:1.7b
   ```

3. Run the local test script:
   ```bash
   cd intro_to_llms
   python local_llm_using_ollama.py
   ```

### Cloud LLM Setup (Groq):
1. Create a `.env` file in the project root:
   ```bash
   echo "GROQ_API_KEY=your_api_key_here" > ../.env
   ```

2. Install dependencies:
   ```bash
   pip install groq python-dotenv
   ```

3. Run the cloud test script:
   ```bash
   cd intro_to_llms
   python cloud_llm_using_groq.py
   ```

If you see responses from both local and cloud models, your environment is ready for more advanced LLM development!
