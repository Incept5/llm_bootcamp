
#!/usr/bin/env python3
"""
Model Comparison Test Script

This script tests various LLM models across different categories:
- Small models (< 10B parameters)
- Medium models (10B - 30B parameters) 
- Large models (> 30B parameters)
- Both local (Ollama) and cloud (Groq) models

Features:
- Performance timing
- Response quality comparison
- Error handling
- Model availability checking
- Detailed logging
"""

import os
import time
import json
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ModelConfig:
    name: str
    provider: str  # 'ollama' or 'groq'
    category: str  # 'small', 'medium', 'large'
    parameters: str  # e.g., '1.7B', '8B', '70B'
    context_length: int
    description: str

@dataclass
class TestResult:
    model_name: str
    provider: str
    response_time: float
    response_text: str
    success: bool
    error_message: Optional[str] = None
    tokens_per_second: Optional[float] = None

class ModelTester:
    def __init__(self):
        self.groq_client = None
        self.ollama_base_url = "http://localhost:11434"
        self.test_prompts = [
            "Hello, how are you?",
            "Explain quantum computing in simple terms.",
            "Write a haiku about coding.",
            "What are the main differences between Python and JavaScript?",
            "Solve this math problem: What is 15% of 240?"
        ]
        
        # Initialize Groq client if API key is available
        if os.environ.get("GROQ_API_KEY"):
            self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
        # Define model configurations
        self.models = [
            # Small Models (< 10B parameters)
            ModelConfig("qwen3:1.7b", "ollama", "small", "1.7B", 32768, "Qwen3 1.7B - Fast, efficient model"),
            ModelConfig("llama3.2:3b", "ollama", "small", "3B", 131072, "Llama 3.2 3B - Good balance of speed and quality"),
            ModelConfig("phi3:3.8b", "ollama", "small", "3.8B", 128000, "Microsoft Phi-3 3.8B - Optimized for reasoning"),
            ModelConfig("gemma2:7b", "ollama", "small", "7B", 8192, "Google Gemma 2 7B - Strong performance"),
            
            # Medium Models (10B - 30B parameters)
            ModelConfig("llama3.1:8b", "ollama", "medium", "8B", 131072, "Llama 3.1 8B - Excellent mid-size model"),
            ModelConfig("qwen2.5:14b", "ollama", "medium", "14B", 131072, "Qwen 2.5 14B - Strong multilingual support"),
            ModelConfig("mixtral:8x7b", "ollama", "medium", "8x7B", 32768, "Mixtral 8x7B - Mixture of Experts architecture"),
            
            # Large Models (> 30B parameters)
            ModelConfig("llama3.1:70b", "ollama", "large", "70B", 131072, "Llama 3.1 70B - High-quality responses"),
            ModelConfig("llama3.3:70b", "ollama", "large", "70B", 131072, "Llama 3.3 70B - Latest version"),
            
            # Cloud Models (Groq)
            ModelConfig("llama-3.3-70b-versatile", "groq", "large", "70B", 131072, "Groq Llama 3.3 70B - Fast cloud inference"),
            ModelConfig("llama-3.1-8b-instant", "groq", "medium", "8B", 131072, "Groq Llama 3.1 8B - Ultra-fast inference"),
            ModelConfig("mistral-saba-24b", "groq", "medium", "8x7B", 32768, "Groq mistral-saba-24b - Efficient MoE"),
            ModelConfig("gemma2-9b-it", "groq", "medium", "9B", 8192, "Groq Gemma 2 9B - Instruction tuned"),
        ]

    def check_ollama_availability(self) -> bool:
        """Check if Ollama server is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def get_ollama_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except requests.exceptions.RequestException:
            return []

    def test_ollama_model(self, model_name: str, prompt: str) -> TestResult:
        """Test a single Ollama model with a prompt"""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                if "error" in data:
                    return TestResult(
                        model_name=model_name,
                        provider="ollama",
                        response_time=response_time,
                        response_text="",
                        success=False,
                        error_message=data["error"]
                    )
                
                response_text = data.get("response", "")
                
                # Calculate tokens per second (rough estimate)
                tokens_per_second = None
                if response_time > 0 and response_text:
                    estimated_tokens = len(response_text.split())
                    tokens_per_second = estimated_tokens / response_time
                
                return TestResult(
                    model_name=model_name,
                    provider="ollama",
                    response_time=response_time,
                    response_text=response_text,
                    success=True,
                    tokens_per_second=tokens_per_second
                )
            else:
                return TestResult(
                    model_name=model_name,
                    provider="ollama",
                    response_time=response_time,
                    response_text="",
                    success=False,
                    error_message=f"HTTP {response.status_code}: {response.text}"
                )
                
        except requests.exceptions.Timeout:
            return TestResult(
                model_name=model_name,
                provider="ollama",
                response_time=60.0,
                response_text="",
                success=False,
                error_message="Request timeout (60s)"
            )
        except Exception as e:
            return TestResult(
                model_name=model_name,
                provider="ollama",
                response_time=time.time() - start_time,
                response_text="",
                success=False,
                error_message=str(e)
            )

    def test_groq_model(self, model_name: str, prompt: str) -> TestResult:
        """Test a single Groq model with a prompt"""
        if not self.groq_client:
            return TestResult(
                model_name=model_name,
                provider="groq",
                response_time=0.0,
                response_text="",
                success=False,
                error_message="Groq API key not configured"
            )
        
        start_time = time.time()
        
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
                timeout=60
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            response_text = chat_completion.choices[0].message.content
            
            # Calculate tokens per second if usage info is available
            tokens_per_second = None
            if hasattr(chat_completion, 'usage') and chat_completion.usage:
                if hasattr(chat_completion.usage, 'completion_tokens') and response_time > 0:
                    tokens_per_second = chat_completion.usage.completion_tokens / response_time
            
            return TestResult(
                model_name=model_name,
                provider="groq",
                response_time=response_time,
                response_text=response_text,
                success=True,
                tokens_per_second=tokens_per_second
            )
            
        except Exception as e:
            return TestResult(
                model_name=model_name,
                provider="groq",
                response_time=time.time() - start_time,
                response_text="",
                success=False,
                error_message=str(e)
            )

    def test_model(self, model_config: ModelConfig, prompt: str) -> TestResult:
        """Test a model based on its configuration"""
        if model_config.provider == "ollama":
            return self.test_ollama_model(model_config.name, prompt)
        elif model_config.provider == "groq":
            return self.test_groq_model(model_config.name, prompt)
        else:
            return TestResult(
                model_name=model_config.name,
                provider=model_config.provider,
                response_time=0.0,
                response_text="",
                success=False,
                error_message=f"Unknown provider: {model_config.provider}"
            )

    def run_comprehensive_test(self, prompt_index: int = 0) -> Dict[str, List[TestResult]]:
        """Run comprehensive test across all models"""
        if prompt_index >= len(self.test_prompts):
            prompt_index = 0
        
        prompt = self.test_prompts[prompt_index]
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE MODEL TEST")
        print(f"{'='*80}")
        print(f"Test Prompt: '{prompt}'")
        print(f"{'='*80}")
        
        # Check system availability
        ollama_available = self.check_ollama_availability()
        groq_available = self.groq_client is not None
        
        print(f"\nSystem Status:")
        print(f"  Ollama Server: {'✓ Available' if ollama_available else '✗ Not available'}")
        print(f"  Groq API: {'✓ Available' if groq_available else '✗ Not configured'}")
        
        if ollama_available:
            available_models = self.get_ollama_models()
            print(f"  Ollama Models: {len(available_models)} available")
        
        results = {
            "small": [],
            "medium": [],
            "large": []
        }
        
        for model_config in self.models:
            # Skip if provider is not available
            if model_config.provider == "ollama" and not ollama_available:
                continue
            if model_config.provider == "groq" and not groq_available:
                continue
            
            # For Ollama models, check if the model is actually available
            if model_config.provider == "ollama":
                available_models = self.get_ollama_models()
                if model_config.name not in available_models:
                    print(f"\n⚠️  Skipping {model_config.name} - not installed")
                    continue
            
            print(f"\n🧪 Testing {model_config.name} ({model_config.provider}) - {model_config.parameters}")
            result = self.test_model(model_config, prompt)
            
            if result.success:
                status = "✅ SUCCESS"
                speed_info = f" ({result.response_time:.2f}s"
                if result.tokens_per_second:
                    speed_info += f", ~{result.tokens_per_second:.1f} tokens/s"
                speed_info += ")"
                print(f"   {status}{speed_info}")
                print(f"   Response: {result.response_text[:100]}{'...' if len(result.response_text) > 100 else ''}")
            else:
                print(f"   ❌ FAILED: {result.error_message}")
            
            results[model_config.category].append(result)
        
        return results

    def generate_summary_report(self, results: Dict[str, List[TestResult]]) -> None:
        """Generate a summary report of test results"""
        print(f"\n{'='*80}")
        print("SUMMARY REPORT")
        print(f"{'='*80}")
        
        for category in ["small", "medium", "large"]:
            category_results = results[category]
            if not category_results:
                continue
            
            print(f"\n{category.upper()} MODELS:")
            print("-" * 40)
            
            successful_results = [r for r in category_results if r.success]
            failed_results = [r for r in category_results if not r.success]
            
            print(f"  Total Tested: {len(category_results)}")
            print(f"  Successful: {len(successful_results)}")
            print(f"  Failed: {len(failed_results)}")
            
            if successful_results:
                avg_response_time = sum(r.response_time for r in successful_results) / len(successful_results)
                print(f"  Average Response Time: {avg_response_time:.2f}s")
                
                # Find fastest model
                fastest = min(successful_results, key=lambda x: x.response_time)
                print(f"  Fastest: {fastest.model_name} ({fastest.response_time:.2f}s)")
                
                # Find slowest model
                slowest = max(successful_results, key=lambda x: x.response_time)
                print(f"  Slowest: {slowest.model_name} ({slowest.response_time:.2f}s)")
            
            if failed_results:
                print(f"  Failed Models:")
                for result in failed_results:
                    print(f"    - {result.model_name}: {result.error_message}")
        
        # Overall statistics
        all_results = []
        for category_results in results.values():
            all_results.extend(category_results)
        
        if all_results:
            successful_count = len([r for r in all_results if r.success])
            total_count = len(all_results)
            success_rate = (successful_count / total_count) * 100
            
            print(f"\nOVERALL STATISTICS:")
            print(f"  Total Models Tested: {total_count}")
            print(f"  Success Rate: {success_rate:.1f}% ({successful_count}/{total_count})")

def main():
    """Main function to run the model comparison tests"""
    tester = ModelTester()
    
    print("LLM Model Comparison Test Suite")
    print("=" * 50)
    
    # Show available test prompts
    print("\nAvailable test prompts:")
    for i, prompt in enumerate(tester.test_prompts):
        print(f"  {i + 1}. {prompt}")
    
    # Get user choice
    try:
        choice = input(f"\nSelect prompt (1-{len(tester.test_prompts)}) or press Enter for default (1): ")
        if choice.strip():
            prompt_index = int(choice) - 1
            if prompt_index < 0 or prompt_index >= len(tester.test_prompts):
                print("Invalid choice, using default prompt.")
                prompt_index = 0
        else:
            prompt_index = 0
    except ValueError:
        print("Invalid input, using default prompt.")
        prompt_index = 0
    
    # Run comprehensive test
    results = tester.run_comprehensive_test(prompt_index)
    
    # Generate summary report
    tester.generate_summary_report(results)
    
    # Save results to JSON file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"model_test_results_{timestamp}.json"
    
    # Convert results to JSON-serializable format
    json_results = {}
    for category, category_results in results.items():
        json_results[category] = []
        for result in category_results:
            json_results[category].append({
                "model_name": result.model_name,
                "provider": result.provider,
                "response_time": result.response_time,
                "response_text": result.response_text,
                "success": result.success,
                "error_message": result.error_message,
                "tokens_per_second": result.tokens_per_second
            })
    
    try:
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "test_prompt": tester.test_prompts[prompt_index],
                "results": json_results
            }, f, indent=2)
        print(f"\n📁 Results saved to: {output_file}")
    except Exception as e:
        print(f"\n⚠️  Could not save results: {e}")

if __name__ == "__main__":
    main()
