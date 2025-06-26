
#!/usr/bin/env python3
"""
Model Size Test Script

Simple script to test and compare different model sizes:
- Small models (fast, less capable)
- Medium models (balanced performance)
- Large models (slow, more capable)

This script focuses on practical comparisons of model sizes
with standardized prompts to evaluate performance vs quality trade-offs.
"""

import os
import time
import requests
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ModelSizeTester:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.groq_client = None
        
        # Initialize Groq if API key available
        if os.environ.get("GROQ_API_KEY"):
            self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
        # Define model categories by size
        self.small_models = [
            {"name": "qwen3:1.7b", "provider": "ollama", "size": "1.7B"},
            {"name": "llama3.2:3b", "provider": "ollama", "size": "3B"},
        ]
        
        self.medium_models = [
            {"name": "llama3.1:8b", "provider": "ollama", "size": "8B"},
            {"name": "llama-3.1-8b-instant", "provider": "groq", "size": "8B"},
            {"name": "gemma2:9b", "provider": "ollama", "size": "9B"},
        ]
        
        self.large_models = [
            {"name": "llama3.1:70b", "provider": "ollama", "size": "70B"},
            {"name": "llama-3.3-70b-versatile", "provider": "groq", "size": "70B"},
        ]
        
        # Test prompts designed to show quality differences
        self.test_prompts = [
            {
                "name": "Simple Greeting",
                "prompt": "Hello! How are you today?",
                "category": "basic"
            },
            {
                "name": "Reasoning Task",
                "prompt": "If I have 5 apples and give away 2, then buy 3 more, how many apples do I have? Explain your reasoning.",
                "category": "reasoning"
            },
            {
                "name": "Creative Writing",
                "prompt": "Write a short poem about the beauty of extras.",
                "category": "creative"
            },
            {
                "name": "Technical Explanation",
                "prompt": "Explain the concept of recursion in programming with a simple example.",
                "category": "technical"
            }
        ]

    def check_ollama_available(self):
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_available_ollama_models(self):
        """Get list of installed Ollama models"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except:
            return []

    def test_ollama_model(self, model_name, prompt):
        """Test an Ollama model"""
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120  # 2 minute timeout for large models
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                if "error" in data:
                    return {"success": False, "error": data["error"], "time": response_time}
                
                return {
                    "success": True,
                    "response": data.get("response", ""),
                    "time": response_time
                }
            else:
                return {"success": False, "error": f"HTTP {response.status_code}", "time": response_time}
                
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Timeout (120s)", "time": 120.0}
        except Exception as e:
            return {"success": False, "error": str(e), "time": time.time() - start_time}

    def test_groq_model(self, model_name, prompt):
        """Test a Groq model"""
        if not self.groq_client:
            return {"success": False, "error": "Groq API key not configured", "time": 0}
        
        start_time = time.time()
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
                timeout=60
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            return {
                "success": True,
                "response": chat_completion.choices[0].message.content,
                "time": response_time
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "time": time.time() - start_time}

    def test_model_category(self, models, category_name, prompt_data):
        """Test all models in a category"""
        print(f"\n{'='*60}")
        print(f"Testing {category_name.upper()} MODELS")
        print(f"Prompt: {prompt_data['prompt']}")
        print(f"{'='*60}")
        
        results = []
        available_ollama_models = self.get_available_ollama_models()
        
        for model in models:
            model_name = model["name"]
            provider = model["provider"]
            size = model["size"]
            
            # Skip unavailable Ollama models
            if provider == "ollama" and model_name not in available_ollama_models:
                print(f"\n❌ {model_name} ({size}) - Not installed")
                continue
            
            # Skip Groq if not configured
            if provider == "groq" and not self.groq_client:
                print(f"\n❌ {model_name} ({size}) - Groq not configured")
                continue
            
            print(f"\n🧪 Testing {model_name} ({size})...")
            
            if provider == "ollama":
                result = self.test_ollama_model(model_name, prompt_data["prompt"])
            else:
                result = self.test_groq_model(model_name, prompt_data["prompt"])
            
            result["model"] = model_name
            result["provider"] = provider
            result["size"] = size
            results.append(result)
            
            if result["success"]:
                print(f"   ✅ Response time: {result['time']:.2f}s")
                print(f"   📝 Response: {result['response'][:100]}{'...' if len(result['response']) > 100 else ''}")
            else:
                print(f"   ❌ Error: {result['error']}")
        
        return results

    def compare_model_sizes(self, prompt_index=0):
        """Compare different model sizes with a specific prompt"""
        if prompt_index >= len(self.test_prompts):
            prompt_index = 0
        
        prompt_data = self.test_prompts[prompt_index]
        
        print(f"\n{'='*80}")
        print("MODEL SIZE COMPARISON TEST")
        print(f"{'='*80}")
        print(f"Test: {prompt_data['name']} ({prompt_data['category']})")
        print(f"{'='*80}")
        
        # Check system status
        ollama_available = self.check_ollama_available()
        groq_available = self.groq_client is not None
        
        print(f"System Status:")
        print(f"  Ollama: {'✅ Available' if ollama_available else '❌ Not available'}")
        print(f"  Groq: {'✅ Available' if groq_available else '❌ Not configured'}")
        
        all_results = []
        
        # Test each category
        if self.small_models:
            small_results = self.test_model_category(self.small_models, "small", prompt_data)
            all_results.extend(small_results)
        
        if self.medium_models:
            medium_results = self.test_model_category(self.medium_models, "medium", prompt_data)
            all_results.extend(medium_results)
        
        if self.large_models:
            large_results = self.test_model_category(self.large_models, "large", prompt_data)
            all_results.extend(large_results)
        
        # Generate comparison summary
        self.generate_size_comparison_summary(all_results, prompt_data)

    def generate_size_comparison_summary(self, results, prompt_data):
        """Generate a summary comparing model sizes"""
        successful_results = [r for r in results if r["success"]]
        
        if not successful_results:
            print(f"\n❌ No successful results to compare")
            return
        
        print(f"\n{'='*80}")
        print("SIZE COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"Test: {prompt_data['name']}")
        print(f"Successful tests: {len(successful_results)}/{len(results)}")
        
        # Sort by response time
        by_speed = sorted(successful_results, key=lambda x: x["time"])
        
        print(f"\n🏃 SPEED RANKING (fastest to slowest):")
        print("-" * 50)
        for i, result in enumerate(by_speed, 1):
            provider_icon = "🌐" if result["provider"] == "groq" else "💻"
            print(f"{i:2d}. {result['model']} ({result['size']}) {provider_icon}")
            print(f"     Time: {result['time']:.2f}s")
        
        print(f"\n📊 SIZE CATEGORY ANALYSIS:")
        print("-" * 50)
        
        # Group by size categories
        small = [r for r in successful_results if "1.7B" in r["size"] or "3B" in r["size"]]
        medium = [r for r in successful_results if "8B" in r["size"] or "9B" in r["size"]]
        large = [r for r in successful_results if "70B" in r["size"]]
        
        categories = [
            ("Small Models (1-3B)", small),
            ("Medium Models (8-9B)", medium),
            ("Large Models (70B)", large)
        ]
        
        for category_name, category_results in categories:
            if category_results:
                avg_time = sum(r["time"] for r in category_results) / len(category_results)
                fastest = min(category_results, key=lambda x: x["time"])
                print(f"\n{category_name}:")
                print(f"  Models tested: {len(category_results)}")
                print(f"  Avg response time: {avg_time:.2f}s")
                print(f"  Fastest: {fastest['model']} ({fastest['time']:.2f}s)")
        
        # Quality assessment (basic - based on response length and content)
        print(f"\n📝 RESPONSE QUALITY INDICATORS:")
        print("-" * 50)
        
        for result in successful_results:
            response_length = len(result["response"])
            quality_indicator = "🔍 Brief" if response_length < 100 else "📄 Detailed" if response_length < 300 else "📚 Comprehensive"
            print(f"{result['model']} ({result['size']}): {quality_indicator} ({response_length} chars)")

def main():
    """Main function"""
    tester = ModelSizeTester()
    
    print("Model Size Comparison Tool")
    print("=" * 40)
    
    # Show available test prompts
    print("\nAvailable test prompts:")
    for i, prompt_data in enumerate(tester.test_prompts):
        print(f"  {i + 1}. {prompt_data['name']} ({prompt_data['category']})")
    
    # Get user choice
    try:
        choice = input(f"\nSelect test (1-{len(tester.test_prompts)}) or Enter for default: ")
        if choice.strip():
            prompt_index = int(choice) - 1
            if prompt_index < 0 or prompt_index >= len(tester.test_prompts):
                print("Invalid choice, using default.")
                prompt_index = 0
        else:
            prompt_index = 0
    except ValueError:
        print("Invalid input, using default.")
        prompt_index = 0
    
    # Run comparison
    tester.compare_model_sizes(prompt_index)

if __name__ == "__main__":
    main()
