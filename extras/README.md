
# Extras - Advanced LLM Features and Capabilities

This directory contains advanced demonstrations and tests that showcase specialized LLM capabilities beyond basic text generation. These examples explore cutting-edge features that enhance the practical utility of local LLMs.

## Files Overview

### 1. `fill_in_middle.py` - Code Completion and Fill-in-the-Middle (FIM)

**What it demonstrates:**
- Fill-in-the-Middle (FIM) capability using code-specialized models
- Code completion and intelligent code generation
- Context-aware programming assistance

**Key features:**
- Uses `qwen2.5-coder` model optimized for coding tasks
- Demonstrates providing both prefix and suffix context to generate middle content
- Shows how LLMs can understand code structure and complete partial implementations
- Uses specialized parameters like `stop` tokens for precise code generation

**Technical details:**
- Model: `qwen2.5-coder` (optimized for code generation)
- Temperature: 0 (deterministic output for code)
- Uses suffix parameter to provide context after the gap
- Demonstrates practical IDE-like code completion functionality

**Use cases:**
- Auto-completion in code editors
- Intelligent code suggestion systems
- Partial code reconstruction
- Template-based code generation

---

### 2. `formatted_output.py` - Structured JSON Response Generation

**What it demonstrates:**
- Constrained generation to produce valid JSON output
- Structured data extraction and formatting
- Multilingual content generation in structured format

**Key features:**
- Forces LLM output to be valid JSON using `format="json"` parameter
- Demonstrates reliable structured data generation
- Shows multilingual capability (English, French, German, Chinese, Russian, Arabic)
- Includes error handling for JSON parsing validation

**Technical details:**
- Model: `qwen2.5-coder` 
- Format constraint: JSON-only output
- Temperature: 0.3 (slight creativity while maintaining structure)
- Extended context window: 8192 tokens
- Includes JSON validation and pretty-printing

**Use cases:**
- Data extraction and transformation
- API response generation
- Structured content creation
- Database record generation
- Configuration file creation

---

### 3. `ollama_function_support.py` - Advanced Function Calling and Tool Integration

**What it demonstrates:**
- Function calling capabilities with local LLMs
- Tool integration and external API simulation
- Comprehensive model benchmarking and testing
- Chain of function calls and complex reasoning

**Key features:**
- **Multiple tool definitions**: Weather, time, location, trigonometry, temperature conversion, Roman numerals
- **Function chaining**: Demonstrates calling multiple functions in sequence
- **Model comparison testing**: Benchmarks different models for function calling accuracy
- **Performance metrics**: Execution time, success rates, statistical analysis
- **Comprehensive test suite**: 10 different test cases with multiple runs per case

**Available Functions:**
- `convert_to_roman_numerals()` - Number to Roman numeral conversion
- `convert_fahrenheit_to_centigrade()` - Temperature conversion
- `day_of_the_week()` - Current day retrieval
- `calc_trig_function()` - Trigonometric calculations
- `weather_tool()` - Weather information (simulated)
- `time_tool()` - Current time (simulated)
- `location_tool()` - Current location (simulated)
- `distance_tool()` - Distance calculations (simulated)

**Testing Framework:**
- Automated model discovery and filtering
- Warmup runs to avoid cold start bias
- Statistical analysis with mean and standard deviation
- CSV export for further analysis
- Detailed success/failure reporting

**Technical details:**
- Comprehensive model compatibility testing
- Temperature: 0.2 for consistent function calling
- Error handling and fallback mechanisms
- Performance benchmarking with timing statistics
- Results visualization with tabulated output

**Use cases:**
- Building AI assistants with external tool access
- Automated workflow systems
- API integration testing
- Model capability assessment
- Function calling reliability testing

## Prerequisites

### System Requirements
- Ollama installed and running locally
- Python 3.8+ with required packages:
  ```bash
  pip install ollama requests tabulate
  ```

### Required Models
- `qwen2.5-coder` - For code completion and JSON generation
- Additional models for function calling tests (automatically discovered)

## Running the Tests

### Code Completion Test
```bash
python fill_in_middle.py
```

### JSON Format Test
```bash
python formatted_output.py
```

### Function Calling Benchmark
```bash
python ollama_function_support.py
```

## Expected Outputs

### Fill-in-the-Middle
Generates the missing code between the prefix and suffix, completing the function implementation.

### Formatted Output
Produces a structured JSON response with numbers 1-10 in multiple languages, demonstrating both multilingual capability and format constraints.

### Function Support
Runs comprehensive tests and produces:
- Real-time test progress with pass/fail indicators
- Statistical summary table
- Detailed performance metrics
- CSV export file (`ollama_function_results.csv`)

## Key Learning Points

1. **Specialized Models**: Code-specific models like `qwen2.5-coder` excel at programming tasks
2. **Format Constraints**: LLMs can be constrained to produce specific output formats reliably
3. **Function Integration**: Modern LLMs can integrate with external tools and APIs
4. **Performance Variability**: Different models have varying capabilities for complex tasks
5. **Benchmarking Importance**: Systematic testing reveals model strengths and limitations

## Advanced Applications

These demonstrations form the foundation for building:
- Code editors with AI assistance
- Structured data processing pipelines
- AI agents with tool access
- Automated testing and validation systems
- Multi-modal AI applications

## Notes

- All tests use local models via Ollama for privacy and control
- Function calling capabilities vary significantly between models
- JSON format constraints ensure reliable structured output
- Code completion works best with code-specialized models
- Performance metrics help in model selection for specific use cases
