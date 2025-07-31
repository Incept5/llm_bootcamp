
# Setup Guide for Ollama Next Token Predictor

This guide will help you set up the Next Token Predictor to use smaller, faster models through Ollama instead of downloading large HuggingFace models.

## Prerequisites

1. **Install Ollama** (if not already installed):
   - Visit [ollama.ai](https://ollama.ai) and download for your platform
   - Or use package managers:
     ```bash
     # macOS
     brew install ollama
     
     # Linux
     curl -fsSL https://ollama.ai/install.sh | sh
     ```

2. **Start Ollama server**:
   ```bash
   ollama serve
   ```

## Recommended Models (Small & Fast)

Choose one of these lightweight models:

### 1. Llama 3.2 1B (Fastest - Recommended)
```bash
ollama pull llama3.2:1b
```
- Size: ~1.3GB
- Parameters: 1 billion
- Best for: Speed and responsiveness

### 2. Phi-3 Mini (Good Quality)
```bash
ollama pull phi3:mini
```
- Size: ~2.2GB  
- Parameters: 3.8 billion
- Best for: Balance of quality and speed

### 3. Qwen 2.5 1.5B (Balanced)
```bash
ollama pull qwen2.5:1.5b
```
- Size: ~1.5GB
- Parameters: 1.5 billion
- Best for: Good quality with reasonable speed

### 4. Llama 3.2 3B (Higher Quality)
```bash
ollama pull llama3.2:3b
```
- Size: ~2.0GB
- Parameters: 3 billion
- Best for: Better text understanding

## Python Dependencies

Install required packages:
```bash
pip install gradio requests numpy matplotlib
```

## Running the Application

1. Make sure Ollama is running:
   ```bash
   ollama serve
   ```

2. Verify your model is available:
   ```bash
   ollama list
   ```

3. Run the predictor:
   ```bash
   python predict_next_token.py
   ```

## Changing Models

To use a different model, edit the `model_name` variable in `predict_next_token.py`:

```python
# In the load_model() function, change this line:
model_name = "llama3.2:1b"  # Change to your preferred model
```

Available options:
- `"llama3.2:1b"` (fastest)
- `"phi3:mini"` (good quality)
- `"qwen2.5:1.5b"` (balanced)
- `"llama3.2:3b"` (higher quality)

## Troubleshooting

### "Ollama server not running"
- Start Ollama: `ollama serve`
- Check if running: `curl http://localhost:11434`

### "Model not available"
- Pull the model: `ollama pull llama3.2:1b`
- List available models: `ollama list`

### Slow responses
- Try a smaller model like `llama3.2:1b`
- Reduce `top_k` parameter in the interface
- Check system resources

## Benefits of This Approach

✅ **Faster downloads**: Models are 1-3GB instead of 7-14GB  
✅ **Less memory usage**: Run efficiently on most systems  
✅ **No GPU required**: Works well on CPU  
✅ **Easy model switching**: Change models without re-downloading  
✅ **Local inference**: All processing stays on your machine  

The token prediction accuracy might be slightly different from larger models, but the interface will be much more responsive and easier to use.
