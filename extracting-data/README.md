
# Data Extraction Demonstrations

This directory contains various Python scripts that demonstrate different techniques for extracting and processing data using Large Language Models (LLMs) and traditional data processing methods.

## Overview

The scripts showcase practical applications of data extraction, from simple text processing to complex database operations and web scraping. Each example demonstrates different LLM providers, data sources, and extraction techniques.

## Scripts and Demonstrations

### 1. Text Extraction with Cloud LLMs - `data_extraction_groq.py`

**Purpose**: Demonstrates text summarization and data extraction using Groq's cloud API with Llama3 70B model.

**Key Features**:
- Uses Groq API for cloud-based LLM processing
- Implements two core functions: `summarise()` and `extract()`
- Processes technical documentation (Magistral-Small-2506 model card)
- Low temperature (0.3) for consistent, focused outputs

**What it demonstrates**:
- Cloud LLM integration for text processing
- Structured prompting for specific extraction tasks
- Processing of technical documentation and specifications

**Dependencies**: `groq`, `python-dotenv`

**Usage**:
```bash
python data_extraction_groq.py
```

**Expected Output**:
- Summary of the technical document
- Extraction of specific information (sampling parameters)

---

### 2. Text Extraction with Local LLMs - `data_extraction_ollama.py`

**Purpose**: Demonstrates the same text processing capabilities using a local Ollama model (Qwen3:4b).

**Key Features**:
- Uses Ollama for local LLM processing
- Same extraction functions as Groq version for comparison
- Error handling for local model availability
- Higher temperature (0.7) with larger context window (8192)

**What it demonstrates**:
- Local vs cloud LLM comparison
- Privacy-focused data processing
- Model parameter tuning for local models

**Dependencies**: `ollama`

**Usage**:
```bash
python data_extraction_ollama.py
```

**Prerequisites**: Ollama installed with qwen3:4b model available

---

### 3. Kaggle Dataset Integration - `load_kaggle.py`

**Purpose**: Demonstrates automated dataset downloading and initial exploration from Kaggle.

**Key Features**:
- Uses `kagglehub` for seamless dataset downloading
- Automatic CSV file detection in downloaded datasets
- Basic data exploration and JSON conversion
- Error handling for missing files

**What it demonstrates**:
- Programmatic access to Kaggle datasets
- Automated data discovery and loading
- Basic data shape analysis and preview

**Dependencies**: `pandas`, `kagglehub`

**Usage**:
```bash
python load_kaggle.py
```

**Expected Output**:
- Dataset download confirmation
- First 6 rows of the Trump tweets dataset
- JSON representation of first 2 rows
- Total row count

---

### 4. Complex Data Processing Pipeline - `payroll.py`

**Purpose**: Comprehensive demonstration of data cleaning, transformation, and SQLite database creation from Kaggle's City of LA payroll data.

**Key Features**:
- Advanced data cleaning functions for monetary and percentage values
- Automated database schema creation with proper indexing
- Statistical analysis and data validation
- Comprehensive error handling for data quality issues

**What it demonstrates**:
- Production-ready data pipeline development
- Data quality assessment and cleaning
- SQL database integration and optimization
- Statistical analysis of large datasets

**Data Processing**:
- Cleans monetary values (removes $, commas, converts to float)
- Processes percentage strings
- Creates SQLite database with optimized indexes
- Generates comprehensive data summaries

**Dependencies**: `pandas`, `numpy`, `sqlite3`, `kagglehub`

**Usage**:
```bash
python payroll.py
```

**Output Files**: Creates `city_payroll.db` SQLite database

---

### 5. LLM-Powered SQL Generation - `payroll2.py`

**Purpose**: Demonstrates using local LLMs to generate SQL queries from natural language descriptions.

**Key Features**:
- Database schema introspection
- Sample data extraction for context
- Natural language to SQL conversion using Ollama
- Query execution and result formatting

**What it demonstrates**:
- Text-to-SQL capabilities of local LLMs
- Context-aware query generation using schema and sample data
- Integration of LLMs with database operations
- Automated result formatting (Markdown tables)

**Dependencies**: `sqlite3`, `requests`, `json`, `re`

**Usage**:
```bash
python payroll2.py
```

**Prerequisites**: 
- `city_payroll.db` database (created by `payroll.py`)
- Ollama running locally with `qwen2.5-coder:latest` model

**Process Flow**:
1. Extracts database schema
2. Retrieves random sample rows for context
3. Sends natural language query to Ollama
4. Parses generated SQL from response
5. Executes query and formats results

---

### 6. Web Scraping with Content Extraction - `scrape.py`

**Purpose**: Sophisticated web scraping focused on extracting structured content from GDPR legal documents.

**Key Features**:
- Intelligent content detection using multiple CSS selectors
- Content filtering to remove navigation and metadata
- Structured text extraction maintaining document hierarchy
- Table processing with format preservation
- Content deduplication and cleaning

**What it demonstrates**:
- Robust web scraping with fallback strategies
- Content structure preservation during extraction
- Legal document processing
- Clean text output generation

**Advanced Techniques**:
- Multiple selector strategies for content detection
- Element filtering based on parent containers and CSS classes
- Heading hierarchy preservation
- Table structure maintenance
- Duplicate content elimination

**Dependencies**: `requests`, `beautifulsoup4`

**Usage**:
```bash
python scrape.py
```

**Output**: Creates `gdpr_article_content.txt` with cleaned, structured content

**Target**: GDPR Article 17 (Right to Erasure) from gdpr-info.eu

## Common Patterns and Techniques

### 1. Error Handling
All scripts implement comprehensive error handling for:
- Network connectivity issues
- Missing dependencies
- Data quality problems
- File system operations

### 2. Data Cleaning
Multiple scripts demonstrate:
- String parsing and normalization
- Numerical data conversion
- Handling missing or invalid values
- Data type validation

### 3. LLM Integration Patterns
- **Cloud vs Local**: Comparison between Groq (cloud) and Ollama (local)
- **Prompt Engineering**: Structured prompts for specific extraction tasks
- **Context Management**: Using schema and sample data for better results
- **Response Processing**: Parsing and validating LLM outputs

### 4. Data Pipeline Development
- **ETL Processes**: Extract, Transform, Load patterns
- **Database Integration**: SQLite for local storage and analysis
- **Statistical Analysis**: Basic descriptive statistics and data profiling

## Getting Started

### Prerequisites
```bash
# Install required packages
pip install pandas numpy requests beautifulsoup4 groq python-dotenv kagglehub ollama

# For Ollama (if using local models)
# Install Ollama and pull required models:
ollama pull qwen3:4b
ollama pull qwen2.5-coder:latest
```

### Environment Setup
Create a `.env` file for API keys:
```
GROQ_API_KEY=your_groq_api_key_here
```

### Execution Order
For full pipeline demonstration:
1. `load_kaggle.py` - Basic dataset loading
2. `payroll.py` - Complex data processing and database creation
3. `payroll2.py` - LLM-powered SQL generation
4. `data_extraction_groq.py` or `data_extraction_ollama.py` - Text extraction
5. `scrape.py` - Web content extraction

## Key Learning Outcomes

1. **LLM Integration**: How to integrate both cloud and local LLMs for data processing tasks
2. **Data Pipeline Design**: Building robust, production-ready data processing pipelines
3. **Error Handling**: Implementing comprehensive error handling in data workflows
4. **Content Extraction**: Various techniques for extracting structured data from different sources
5. **Database Operations**: Using databases effectively in data processing workflows
6. **Web Scraping**: Sophisticated content extraction from web sources

## Notes

- All scripts include extensive error handling and validation
- Database operations use SQLite for simplicity and portability
- LLM prompts are designed for specific, focused tasks
- Code includes comprehensive documentation and comments
- Scripts demonstrate both simple and complex data processing patterns

## Troubleshooting

**Common Issues**:
- **Ollama Connection**: Ensure Ollama is running locally on port 11434
- **Model Availability**: Check that required models are downloaded (`ollama list`)
- **API Keys**: Verify Groq API key is properly set in environment
- **Dataset Access**: Ensure Kaggle credentials are configured for dataset downloads
- **Database Permissions**: Check write permissions for SQLite database creation
