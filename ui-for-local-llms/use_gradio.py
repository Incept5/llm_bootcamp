#!/usr/bin/env python3
"""
Gradio UI for Local LLMs (Ollama)

This script creates a web-based user interface using Gradio that allows users to:
- List and select from available Ollama models
- Enter text prompts
- Receive streaming responses from the selected LLM

Features:
- Model auto-discovery from Ollama
- Real-time streaming responses
- Clean, responsive UI
- Error handling and status feedback
"""

import gradio as gr
import requests
import json
import time
from typing import List, Generator, Optional

class OllamaUI:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.available_models = []
        self.refresh_models()
    
    def check_ollama_status(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                return sorted(models)
            return []
        except requests.exceptions.RequestException:
            return []
    
    def refresh_models(self) -> List[str]:
        """Refresh the list of available models"""
        if self.check_ollama_status():
            self.available_models = self.get_available_models()
            if not self.available_models:
                self.available_models = ["No models found"]
        else:
            self.available_models = ["Ollama not available"]
        return self.available_models
    
    def stream_response(self, model_name: str, prompt: str, system_prompt: str = "") -> Generator[str, None, None]:
        """Stream response from selected model"""
        # Input validation
        if not prompt.strip():
            yield "❌ **Error**: Please enter a prompt"
            return
        
        if model_name in ["No models found", "Ollama not available"]:
            yield "❌ **Error**: Please select a valid model"
            return
        
        # Check Ollama status
        if not self.check_ollama_status():
            yield "❌ **Error**: Ollama server is not running. Please start Ollama with `ollama serve`"
            return
        
        # Show processing message
        yield "🤖 **Processing your request...** \n\nConnecting to model and generating response..."
        
        try:
            # Prepare the request
            request_data = {
                "model": model_name,
                "prompt": prompt,
                "stream": True
            }
            
            # Add system prompt if provided
            if system_prompt.strip():
                request_data["system"] = system_prompt.strip()
            
            # Make streaming request
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=request_data,
                timeout=120,
                stream=True
            )
            
            if response.status_code != 200:
                yield f"❌ **Error**: HTTP {response.status_code} - {response.text}"
                return
            
            # Process streaming response
            accumulated_response = ""
            first_chunk = True
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        
                        # Check for errors
                        if "error" in chunk:
                            yield f"❌ **Error**: {chunk['error']}"
                            return
                        
                        # Process response content
                        if "response" in chunk:
                            if first_chunk:
                                # Clear the processing message and start with actual response
                                accumulated_response = chunk["response"]
                                first_chunk = False
                            else:
                                accumulated_response += chunk["response"]
                            
                            yield accumulated_response
                        
                        # Check if done
                        if chunk.get("done", False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
            
            # If no response was generated
            if first_chunk:
                yield "❌ **Error**: No response generated from the model"
                
        except requests.exceptions.Timeout:
            yield "❌ **Error**: Request timed out (120s). The model might be too large or busy."
        except requests.exceptions.RequestException as e:
            yield f"❌ **Error**: Network error - {str(e)}"
        except Exception as e:
            yield f"❌ **Error**: Unexpected error - {str(e)}"
    
    def get_model_info(self, model_name: str) -> str:
        """Get information about the selected model"""
        if model_name in ["No models found", "Ollama not available", ""]:
            return "Select a model to see information"
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/show",
                json={"name": model_name},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract key information
                info_parts = []
                
                # Model details
                if "details" in data:
                    details = data["details"]
                    if "family" in details:
                        info_parts.append(f"**Family**: {details['family']}")
                    if "parameter_size" in details:
                        info_parts.append(f"**Parameters**: {details['parameter_size']}")
                    if "quantization_level" in details:
                        info_parts.append(f"**Quantization**: {details['quantization_level']}")
                
                # Model file info
                if "model_info" in data:
                    model_info = data["model_info"]
                    for key, value in model_info.items():
                        if key in ["general.architecture", "general.name"]:
                            info_parts.append(f"**{key.split('.')[-1].title()}**: {value}")
                
                if info_parts:
                    return "\n".join(info_parts)
                else:
                    return f"**Model**: {model_name}\n*Additional details not available*"
            else:
                return f"**Model**: {model_name}\n*Could not fetch model details*"
                
        except Exception as e:
            return f"**Model**: {model_name}\n*Error fetching details: {str(e)}*"

def create_interface():
    """Create and configure the Gradio interface"""
    
    ollama_ui = OllamaUI()
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .model-info {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(css=css, title="Ollama LLM Interface") as interface:
        gr.Markdown("""
        # 🚀 Ollama LLM Interface
        
        Interactive interface for local LLMs using Ollama. Select a model, enter your prompt, and receive streaming responses.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Model selection and info
                gr.Markdown("### 🤖 Model Selection")
                
                model_dropdown = gr.Dropdown(
                    choices=ollama_ui.available_models,
                    value=ollama_ui.available_models[0] if ollama_ui.available_models else None,
                    label="Select Model",
                    interactive=True
                )
                
                refresh_btn = gr.Button("🔄 Refresh Models", variant="secondary", size="sm")
                
                model_info = gr.Markdown(
                    value=ollama_ui.get_model_info(ollama_ui.available_models[0] if ollama_ui.available_models else ""),
                    elem_classes=["model-info"]
                )
                
                # System prompt (optional)
                gr.Markdown("### ⚙️ System Prompt (Optional)")
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    placeholder="Enter system instructions to customize the AI's behavior...",
                    lines=3,
                    value="You are a helpful AI assistant. Provide clear, accurate, and helpful responses."
                )
                
            with gr.Column(scale=2):
                # Chat interface
                gr.Markdown("### 💬 Chat Interface")
                
                user_input = gr.Textbox(
                    label="Your Prompt",
                    placeholder="Enter your question or prompt here...",
                    lines=4
                )
                
                submit_btn = gr.Button("🚀 Send", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="sm")
                
                response_output = gr.Markdown(
                    label="AI Response",
                    value="Select a model and enter a prompt to get started!",
                    height=400
                )
        
        # Status information
        with gr.Row():
            gr.Markdown("""
            ### 📋 Usage Tips
            - **Model Selection**: Choose from available Ollama models
            - **System Prompt**: Optional instructions to customize AI behavior
            - **Streaming**: Responses appear in real-time as they're generated
            - **Refresh Models**: Click to update the model list if you've installed new models
            
            ### 🛠️ Troubleshooting
            - Ensure Ollama is running: `ollama serve`
            - Check available models: `ollama list`
            - Install models: `ollama pull <model-name>`
            """)
        
        # Event handlers
        def refresh_models_handler():
            new_models = ollama_ui.refresh_models()
            return gr.Dropdown(choices=new_models, value=new_models[0] if new_models else None)
        
        def update_model_info(model_name):
            return ollama_ui.get_model_info(model_name)
        
        def clear_interface():
            return "", "Select a model and enter a prompt to get started!"
        
        def handle_submit(model_name, prompt, system_prompt):
            return ollama_ui.stream_response(model_name, prompt, system_prompt)
        
        # Wire up events
        refresh_btn.click(
            fn=refresh_models_handler,
            outputs=[model_dropdown]
        )
        
        model_dropdown.change(
            fn=update_model_info,
            inputs=[model_dropdown],
            outputs=[model_info]
        )
        
        submit_btn.click(
            fn=handle_submit,
            inputs=[model_dropdown, user_input, system_prompt],
            outputs=[response_output]
        )
        
        user_input.submit(
            fn=handle_submit,
            inputs=[model_dropdown, user_input, system_prompt],
            outputs=[response_output]
        )
        
        clear_btn.click(
            fn=clear_interface,
            outputs=[user_input, response_output]
        )
    
    return interface

def main():
    """Main function to launch the interface"""
    print("🚀 Starting Ollama LLM Interface...")
    print("=" * 50)
    
    # Create UI instance to check status
    ui = OllamaUI()
    
    # Check Ollama status
    if ui.check_ollama_status():
        models = ui.get_available_models()
        print(f"✅ Ollama server is running")
        print(f"📦 Found {len(models)} models: {', '.join(models) if models else 'None'}")
    else:
        print("❌ Ollama server is not running")
        print("💡 Please start Ollama with: ollama serve")
        print("💡 Or install models with: ollama pull <model-name>")
    
    print("\n🌐 Starting Gradio interface...")
    
    # Create and launch interface
    interface = create_interface()
    
    interface.launch(
        inbrowser=True,           # Open browser automatically
        share=False,             # Don't create public link
        server_name="localhost",  # Run on localhost
        server_port=7860,        # Default Gradio port
        show_api=False,          # Don't show API docs
        quiet=False              # Show startup messages
    )

if __name__ == "__main__":
    main()
