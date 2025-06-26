#!/usr/bin/env python3
"""
Test script to isolate the urllib3/LibreSSL issue
"""

print("Testing imports...")

try:
    import gradio as gr
    print("✓ gradio imported successfully")
except Exception as e:
    print(f"✗ gradio import failed: {e}")

try:
    import requests
    print("✓ requests imported successfully")
except Exception as e:
    print(f"✗ requests import failed: {e}")

try:
    import urllib3
    print(f"✓ urllib3 imported successfully (version: {urllib3.__version__})")
except Exception as e:
    print(f"✗ urllib3 import failed: {e}")

print("\nTesting Ollama connection...")
try:
    response = requests.get("http://localhost:11434/api/tags", timeout=5)
    print(f"✓ Ollama connection successful (status: {response.status_code})")
except requests.exceptions.ConnectionError:
    print("✗ Ollama connection failed - server not running")
except Exception as e:
    print(f"✗ Ollama connection failed: {e}")

print("\nTesting minimal Gradio interface...")
try:
    def test_function(text):
        return f"Echo: {text}"
    
    # Create a simple interface but don't launch it
    demo = gr.Interface(
        fn=test_function,
        inputs="text",
        outputs="text"
    )
    print("✓ Gradio interface created successfully")
    
    # Try to launch with share=False and server_name=None to avoid network issues
    print("Testing Gradio launch...")
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860, prevent_thread_lock=True)
    print("✓ Gradio launched successfully")
    
except Exception as e:
    print(f"✗ Gradio interface failed: {e}")

print("\nAll tests completed!")
