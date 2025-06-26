2
2
#!/usr/bin/env python3
"""
Streaming Performance Test Script

This script tests streaming vs non-streaming response performance
and measures time-to-first-token (TTFT) and tokens per second for
different models and configurations.

Key metrics:
- Time to First Token (TTFT) - Latency measure
- Tokens per Second - Throughput measure
- Total Response Time
- Response Quality
"""

import os
import time
import json
import requests
from typing import Dict, List, Optional, Generator
from dataclasses import dataclass
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class StreamingResult:
    model_name: str
    provider: str
    streaming: bool
    time_to_first_token: Optional[float]
    tokens_per_second: Optional[float]
    total_time: float
    total_tokens: int
    response_text: str
    success: bool
    error_message: Optional[str] = None

class StreamingTester:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.groq_client = None
        
        if os.environ.get("GROQ_API_KEY"):
            self.groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
        # Test models optimized for streaming
        self.test_models = [
            {"name": "llama3.2:3b", "provider": "ollama"},
            {"name": "llama3.1:8b", "provider": "ollama"},
            {"name": "qwen3:1.7b", "provider": "ollama"},
            {"name": "llama-3.1-8b-instant", "provider": "groq"},
            {"name": "llama-3.3-70b-versatile", "provider": "groq"},
        ]
        
        # Test prompts of varying lengths to test streaming behavior
        self.test_prompts = [
            {
                "name": "Short Response",
                "prompt": "Say hello in 3 different languages.",
                "expected_length": "short"
            },
            {
                "name": "Medium Response", 
                "prompt": "Explain the concept of machine learning in simple terms with examples.",
                "expected_length": "medium"
            },
            {
                "name": "Long Response",
                "prompt": "Write a detailed guide on getting started with Python programming, including installation, basic syntax, and a simple project example.",
                "expected_length": "long"
            },
            {
                "name": "Code Generation",
                "prompt": "Write a Python function that reads a CSV file, processes the data, and creates a simple visualization. Include comments and error handling.",
                "expected_length": "extras"
            }
        ]

    def count_tokens_estimate(self, text: str) -> int:
        """Rough token count estimation (1 token ≈ 0.75 words)"""
        return max(1, int(len(text.split()) * 1.33))

    def test_ollama_streaming(self, model_name: str, prompt: str) -> StreamingResult:
        """Test Ollama model with streaming enabled"""
        start_time = time.time()
        first_token_time = None
        response_text = ""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": True
                },
                timeout=120,
                stream=True
            )
            
            if response.status_code != 200:
                return StreamingResult(
                    model_name=model_name,
                    provider="ollama",
                    streaming=True,
                    time_to_first_token=None,
                    tokens_per_second=None,
                    total_time=time.time() - start_time,
                    total_tokens=0,
                    response_text="",
                    success=False,
                    error_message=f"HTTP {response.status_code}"
                )
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "error" in chunk:
                            return StreamingResult(
                                model_name=model_name,
                                provider="ollama",
                                streaming=True,
                                time_to_first_token=first_token_time,
                                tokens_per_second=None,
                                total_time=time.time() - start_time,
                                total_tokens=0,
                                response_text=response_text,
                                success=False,
                                error_message=chunk["error"]
                            )
                        
                        if "response" in chunk:
                            if first_token_time is None:
                                first_token_time = time.time() - start_time
                            response_text += chunk["response"]
                        
                        if chunk.get("done", False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
            
            end_time = time.time()
            total_time = end_time - start_time
            total_tokens = self.count_tokens_estimate(response_text)
            
            tokens_per_second = None
            if total_time > 0 and total_tokens > 0:
                tokens_per_second = total_tokens / total_time
            
            return StreamingResult(
                model_name=model_name,
                provider="ollama",
                streaming=True,
                time_to_first_token=first_token_time,
                tokens_per_second=tokens_per_second,
                total_time=total_time,
                total_tokens=total_tokens,
                response_text=response_text,
                success=True
            )
            
        except Exception as e:
            return StreamingResult(
                model_name=model_name,
                provider="ollama",
                streaming=True,
                time_to_first_token=first_token_time,
                tokens_per_second=None,
                total_time=time.time() - start_time,
                total_tokens=0,
                response_text=response_text,
                success=False,
                error_message=str(e)
            )

    def test_ollama_non_streaming(self, model_name: str, prompt: str) -> StreamingResult:
        """Test Ollama model without streaming"""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=120
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                if "error" in data:
                    return StreamingResult(
                        model_name=model_name,
                        provider="ollama",
                        streaming=False,
                        time_to_first_token=total_time,  # Same as total for non-streaming
                        tokens_per_second=None,
                        total_time=total_time,
                        total_tokens=0,
                        response_text="",
                        success=False,
                        error_message=data["error"]
                    )
                
                response_text = data.get("response", "")
                total_tokens = self.count_tokens_estimate(response_text)
                
                tokens_per_second = None
                if total_time > 0 and total_tokens > 0:
                    tokens_per_second = total_tokens / total_time
                
                return StreamingResult(
                    model_name=model_name,
                    provider="ollama",
                    streaming=False,
                    time_to_first_token=total_time,
                    tokens_per_second=tokens_per_second,
                    total_time=total_time,
                    total_tokens=total_tokens,
                    response_text=response_text,
                    success=True
                )
            else:
                return StreamingResult(
                    model_name=model_name,
                    provider="ollama",
                    streaming=False,
                    time_to_first_token=None,
                    tokens_per_second=None,
                    total_time=total_time,
                    total_tokens=0,
                    response_text="",
                    success=False,
                    error_message=f"HTTP {response.status_code}"
                )
                
        except Exception as e:
            return StreamingResult(
                model_name=model_name,
                provider="ollama",
                streaming=False,
                time_to_first_token=None,
                tokens_per_second=None,
                total_time=time.time() - start_time,
                total_tokens=0,
                response_text="",
                success=False,
                error_message=str(e)
            )

    def test_groq_streaming(self, model_name: str, prompt: str) -> StreamingResult:
        """Test Groq model with streaming enabled"""
        if not self.groq_client:
            return StreamingResult(
                model_name=model_name,
                provider="groq",
                streaming=True,
                time_to_first_token=None,
                tokens_per_second=None,
                total_time=0,
                total_tokens=0,
                response_text="",
                success=False,
                error_message="Groq API key not configured"
            )
        
        start_time = time.time()
        first_token_time = None
        response_text = ""
        
        try:
            stream = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
                stream=True,
                timeout=60
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    response_text += chunk.choices[0].delta.content
            
            end_time = time.time()
            total_time = end_time - start_time
            total_tokens = self.count_tokens_estimate(response_text)
            
            tokens_per_second = None
            if total_time > 0 and total_tokens > 0:
                tokens_per_second = total_tokens / total_time
            
            return StreamingResult(
                model_name=model_name,
                provider="groq",
                streaming=True,
                time_to_first_token=first_token_time,
                tokens_per_second=tokens_per_second,
                total_time=total_time,
                total_tokens=total_tokens,
                response_text=response_text,
                success=True
            )
            
        except Exception as e:
            return StreamingResult(
                model_name=model_name,
                provider="groq",
                streaming=True,
                time_to_first_token=first_token_time,
                tokens_per_second=None,
                total_time=time.time() - start_time,
                total_tokens=0,
                response_text=response_text,
                success=False,
                error_message=str(e)
            )

    def test_groq_non_streaming(self, model_name: str, prompt: str) -> StreamingResult:
        """Test Groq model without streaming"""
        if not self.groq_client:
            return StreamingResult(
                model_name=model_name,
                provider="groq",
                streaming=False,
                time_to_first_token=None,
                tokens_per_second=None,
                total_time=0,
                total_tokens=0,
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
            total_time = end_time - start_time
            
            response_text = chat_completion.choices[0].message.content
            total_tokens = self.count_tokens_estimate(response_text)
            
            tokens_per_second = None
            if total_time > 0 and total_tokens > 0:
                tokens_per_second = total_tokens / total_time
            
            return StreamingResult(
                model_name=model_name,
                provider="groq",
                streaming=False,
                time_to_first_token=total_time,  # Same as total for non-streaming
                tokens_per_second=tokens_per_second,
                total_time=total_time,
                total_tokens=total_tokens,
                response_text=response_text,
                success=True
            )
            
        except Exception as e:
            return StreamingResult(
                model_name=model_name,
                provider="groq",
                streaming=False,
                time_to_first_token=None,
                tokens_per_second=None,
                total_time=time.time() - start_time,
                total_tokens=0,
                response_text="",
                success=False,
                error_message=str(e)
            )

    def run_streaming_comparison(self, prompt_index: int = 0) -> Dict[str, List[StreamingResult]]:
        """Run comprehensive streaming vs non-streaming comparison"""
        if prompt_index >= len(self.test_prompts):
            prompt_index = 0
        
        prompt_data = self.test_prompts[prompt_index]
        prompt = prompt_data["prompt"]
        
        print(f"\n{'='*80}")
        print("STREAMING PERFORMANCE COMPARISON")
        print(f"{'='*80}")
        print(f"Test: {prompt_data['name']} (Expected: {prompt_data['expected_length']})")
        print(f"Prompt: {prompt}")
        print(f"{'='*80}")
        
        # Check availability
        ollama_available = self.check_ollama_available()
        groq_available = self.groq_client is not None
        
        print(f"System Status:")
        print(f"  Ollama: {'✅' if ollama_available else '❌'}")
        print(f"  Groq: {'✅' if groq_available else '❌'}")
        
        results = {
            "streaming": [],
            "non_streaming": []
        }
        
        available_ollama_models = self.get_available_ollama_models() if ollama_available else []
        
        for model in self.test_models:
            model_name = model["name"]
            provider = model["provider"]
            
            # Skip unavailable models
            if provider == "ollama":
                if not ollama_available or model_name not in available_ollama_models:
                    print(f"\n❌ Skipping {model_name} - not available")
                    continue
            elif provider == "groq":
                if not groq_available:
                    print(f"\n❌ Skipping {model_name} - Groq not configured")
                    continue
            
            print(f"\n🧪 Testing {model_name} ({provider})")
            
            # Test streaming
            print("   📡 Testing streaming...")
            if provider == "ollama":
                streaming_result = self.test_ollama_streaming(model_name, prompt)
            else:
                streaming_result = self.test_groq_streaming(model_name, prompt)
            
            results["streaming"].append(streaming_result)
            
            if streaming_result.success:
                ttft = streaming_result.time_to_first_token or 0
                tps = streaming_result.tokens_per_second or 0
                print(f"      ✅ TTFT: {ttft:.3f}s, TPS: {tps:.1f}, Total: {streaming_result.total_time:.2f}s")
            else:
                print(f"      ❌ Error: {streaming_result.error_message}")
            
            # Test non-streaming
            print("   📄 Testing non-streaming...")
            if provider == "ollama":
                non_streaming_result = self.test_ollama_non_streaming(model_name, prompt)
            else:
                non_streaming_result = self.test_groq_non_streaming(model_name, prompt)
            
            results["non_streaming"].append(non_streaming_result)
            
            if non_streaming_result.success:
                tps = non_streaming_result.tokens_per_second or 0
                print(f"      ✅ Total: {non_streaming_result.total_time:.2f}s, TPS: {tps:.1f}")
            else:
                print(f"      ❌ Error: {non_streaming_result.error_message}")
        
        return results

    def check_ollama_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def get_available_ollama_models(self) -> List[str]:
        """Get available Ollama models"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except:
            return []

    def generate_streaming_report(self, results: Dict[str, List[StreamingResult]], prompt_data: dict):
        """Generate detailed streaming performance report"""
        print(f"\n{'='*80}")
        print("STREAMING PERFORMANCE REPORT")
        print(f"{'='*80}")
        
        streaming_results = [r for r in results["streaming"] if r.success]
        non_streaming_results = [r for r in results["non_streaming"] if r.success]
        
        if not streaming_results and not non_streaming_results:
            print("❌ No successful results to analyze")
            return
        
        print(f"\n📊 PERFORMANCE METRICS COMPARISON")
        print("-" * 60)
        print(f"{'Model':<25} {'Mode':<12} {'TTFT':<8} {'TPS':<8} {'Total':<8}")
        print("-" * 60)
        
        # Combine and sort results for comparison
        all_results = []
        for result in streaming_results:
            all_results.append(("Streaming", result))
        for result in non_streaming_results:
            all_results.append(("Non-Stream", result))
        
        # Sort by model name for easier comparison
        all_results.sort(key=lambda x: x[1].model_name)
        
        for mode, result in all_results:
            ttft = f"{result.time_to_first_token:.3f}s" if result.time_to_first_token else "N/A"
            tps = f"{result.tokens_per_second:.1f}" if result.tokens_per_second else "N/A"
            total = f"{result.total_time:.2f}s"
            
            print(f"{result.model_name:<25} {mode:<12} {ttft:<8} {tps:<8} {total:<8}")
        
        # Analysis by model
        print(f"\n🔍 MODEL-BY-MODEL ANALYSIS")
        print("-" * 60)
        
        models_tested = set()
        for result_list in results.values():
            for result in result_list:
                if result.success:
                    models_tested.add(result.model_name)
        
        for model_name in sorted(models_tested):
            streaming_result = next((r for r in streaming_results if r.model_name == model_name), None)
            non_streaming_result = next((r for r in non_streaming_results if r.model_name == model_name), None)
            
            print(f"\n{model_name}:")
            
            if streaming_result and non_streaming_result:
                # Compare streaming vs non-streaming
                ttft_advantage = streaming_result.time_to_first_token or 0
                total_time_diff = streaming_result.total_time - non_streaming_result.total_time
                
                print(f"  Time to First Token: {ttft_advantage:.3f}s (streaming advantage)")
                print(f"  Total Time Difference: {total_time_diff:+.2f}s ({'streaming faster' if total_time_diff < 0 else 'non-streaming faster'})")
                
                if streaming_result.tokens_per_second and non_streaming_result.tokens_per_second:
                    tps_diff = streaming_result.tokens_per_second - non_streaming_result.tokens_per_second
                    print(f"  TPS Difference: {tps_diff:+.1f} ({'streaming faster' if tps_diff > 0 else 'non-streaming faster'})")
            
            elif streaming_result:
                print(f"  Streaming only - TTFT: {streaming_result.time_to_first_token:.3f}s, TPS: {streaming_result.tokens_per_second:.1f}")
            elif non_streaming_result:
                print(f"  Non-streaming only - Total: {non_streaming_result.total_time:.2f}s, TPS: {non_streaming_result.tokens_per_second:.1f}")
        
        # Overall insights
        print(f"\n💡 KEY INSIGHTS")
        print("-" * 40)
        
        if streaming_results:
            avg_ttft = sum(r.time_to_first_token for r in streaming_results if r.time_to_first_token) / len(streaming_results)
            fastest_ttft = min((r for r in streaming_results if r.time_to_first_token), key=lambda x: x.time_to_first_token)
            print(f"• Average Time to First Token: {avg_ttft:.3f}s")
            print(f"• Fastest TTFT: {fastest_ttft.model_name} ({fastest_ttft.time_to_first_token:.3f}s)")
        
        if streaming_results and non_streaming_results:
            streaming_avg = sum(r.total_time for r in streaming_results) / len(streaming_results)
            non_streaming_avg = sum(r.total_time for r in non_streaming_results) / len(non_streaming_results)
            print(f"• Streaming avg total time: {streaming_avg:.2f}s")
            print(f"• Non-streaming avg total time: {non_streaming_avg:.2f}s")
            print(f"• Overall advantage: {'Streaming' if streaming_avg < non_streaming_avg else 'Non-streaming'}")

def main():
    """Main function"""
    tester = StreamingTester()
    
    print("Streaming Performance Test Suite")
    print("=" * 40)
    
    # Show available test prompts
    print("\nAvailable test prompts:")
    for i, prompt_data in enumerate(tester.test_prompts):
        print(f"  {i + 1}. {prompt_data['name']} ({prompt_data['expected_length']})")
    
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
    
    # Run streaming comparison
    results = tester.run_streaming_comparison(prompt_index)
    
    # Generate report
    tester.generate_streaming_report(results, tester.test_prompts[prompt_index])

if __name__ == "__main__":
    main()
