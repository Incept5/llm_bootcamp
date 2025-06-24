
#!/usr/bin/env python3
"""
Ollama-based Grimm Fairy Tales RAG Search Demo

A minimal RAG system using Ollama embeddings for demonstration purposes.
Equivalent to qwen3-embedding-demo.py but using Ollama models throughout.
"""

import numpy as np
import requests
import time
import re
from typing import List, Dict, Any, Optional, Tuple
import ollama
from ollama import Options
from rich.console import Console
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


# Configuration
EMBEDDING_MODEL = 'nomic-embed-text'  # 768 dimensions
OLLAMA_BASE_URL = 'http://localhost:11434'
BATCH_SIZE = 10  # Smaller batches for API calls
MAX_TOKEN_LENGTH = 512
TOP_K = 6
MAX_WORKERS = 4  # For concurrent API calls


def get_ollama_embedding(text: str, model: str = EMBEDDING_MODEL) -> Optional[np.ndarray]:
    """Get embedding from Ollama API for a single text."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30
        )
        response.raise_for_status()
        embedding = np.array(response.json()["embedding"])
        # Normalize the embedding
        return embedding / np.linalg.norm(embedding)
    except Exception as e:
        print(f"Error getting embedding for text: {e}")
        return None


def get_ollama_embeddings_batch(texts: List[str], model: str = EMBEDDING_MODEL) -> List[Optional[np.ndarray]]:
    """Get embeddings for multiple texts with concurrent processing."""
    console = Console()
    
    def get_single_embedding(text):
        return get_ollama_embedding(text, model)
    
    embeddings = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_index = {executor.submit(get_single_embedding, text): i 
                          for i, text in enumerate(texts)}
        
        # Collect results in order
        results = [None] * len(texts)
        completed = 0
        
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
                completed += 1
                if completed % 5 == 0:  # Progress indicator
                    console.print(f"[dim]Processed {completed}/{len(texts)} embeddings...[/dim]")
            except Exception as e:
                console.print(f"[red]Error processing embedding {index}: {e}[/red]")
                results[index] = None
    
    return results


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two normalized vectors."""
    return np.dot(a, b)


def chunk_text(text: str) -> List[Dict[str, Any]]:
    """Split text into chapters using quadruple newlines."""
    chapters = re.split(r'\n\n\n\n', text)
    
    # Initialize tiktoken encoder
    encoder = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
    
    chunks = []
    for i, chapter in enumerate(chapters):
        cleaned = ' '.join(chapter.strip().split())
        if len(cleaned) > 100:  # Only include substantial chapters
            token_count = len(encoder.encode(cleaned))
            chunks.append({
                'id': i,
                'text': cleaned,
                'length': len(cleaned),
                'tokens': token_count
            })
    
    return chunks


def search_chunks(query: str, chunks: List[Dict[str, Any]], chunk_embeddings: List[np.ndarray]) -> List[Dict[str, Any]]:
    """Search chunks using embedding similarity."""
    console = Console()
    
    # Get query embedding
    console.print(f"[dim]Getting embedding for query...[/dim]")
    query_embedding = get_ollama_embedding(query)
    if query_embedding is None:
        console.print("[red]Failed to get query embedding![/red]")
        return []
    
    # Compute similarities
    similarities = []
    valid_chunks = []
    
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        if embedding is not None:
            similarity = cosine_similarity(query_embedding, embedding)
            similarities.append((similarity, len(valid_chunks)))
            valid_chunks.append(chunk)
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    # Build results
    results = []
    for i in range(min(TOP_K, len(similarities))):
        similarity_score, chunk_idx = similarities[i]
        chunk = valid_chunks[chunk_idx]
        results.append({
            'text': chunk['text'],
            'similarity': similarity_score,
            'length': chunk['length'],
            'tokens': chunk['tokens']
        })
    
    return results


def generate_all_embeddings(chunks: List[Dict[str, Any]]) -> List[Optional[np.ndarray]]:
    """Generate embeddings for all chunks using Ollama."""
    console = Console()
    console.print(f"[bold]Generating embeddings for {len(chunks)} chunks using {EMBEDDING_MODEL}...[/bold]")
    
    # Process in batches
    all_embeddings = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        batch_texts = [chunk['text'][:2000] for chunk in batch_chunks]  # Truncate very long texts
        
        console.print(f"[dim]Processing batch {i//BATCH_SIZE + 1}/{(len(chunks)-1)//BATCH_SIZE + 1}...[/dim]")
        batch_embeddings = get_ollama_embeddings_batch(batch_texts)
        all_embeddings.extend(batch_embeddings)
    
    # Check for failed embeddings
    failed_count = sum(1 for emb in all_embeddings if emb is None)
    if failed_count > 0:
        console.print(f"[yellow]Warning: {failed_count} embeddings failed to generate[/yellow]")
    
    return all_embeddings


def display_results(query: str, results: List[Dict[str, Any]]) -> None:
    """Display search results."""
    console = Console()
    
    console.print(f"\n[bold]Query:[/bold] '{query}'")
    console.print("=" * 50)
    
    total_tokens = 0
    for i, result in enumerate(results, 1):
        similarity = result['similarity']
        text = result['text']
        tokens = result['tokens']
        total_tokens += tokens
        display_text = text[:150] + "..." if len(text) > 150 else text
        console.print(f"[bold]{i}.[/bold] [green][Score: {similarity:.3f}][/green] [blue][Tokens: {tokens}][/blue] {display_text}")
    
    console.print(f"\n[bold]Total tokens in results:[/bold] {total_tokens}")
    console.print()


def generate_answer(query: str, results: List[Dict[str, Any]]) -> None:
    """Generate a complete answer using qwen3 via ollama with retrieved context."""
    LLM = "qwen3:4b"
    THINKING = False
    console = Console()
    
    # Format context from retrieved chunks
    context = "\n\n".join([f"Context {i+1}: {result['text'][:1000]}..." if len(result['text']) > 1000 
                           else f"Context {i+1}: {result['text']}" 
                           for i, result in enumerate(results[:3])])  # Use top 3 results
    
    system_prompt = """You are a helpful assistant that answers questions about Grimm fairy tales. 
    Use the provided context from the fairy tales to answer the user's question accurately and concisely.
    Provide your answer in English, even though the source text may be in German.
    Keep your response short and focused on directly answering the question."""
    
    instruction = f"""Based on the following context from Grimm fairy tales, please answer this question: {query}

Context:
{context}

Please provide a short, direct answer in English."""
    
    try:
        console.print("[dim]Generating answer with qwen3...[/dim]")
        response = ollama.chat(
            model=LLM,
            think=THINKING,
            stream=False,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': instruction}
            ],
            options=Options(
                temperature=0.7,
                num_ctx=32768,  # 32k context
                top_p=0.95,
                top_k=40,
                num_predict=-1
            )
        )
        
        console.print("\n[bold green]🧠 RAG Answer:[/bold green]")
        
        if hasattr(response.message, 'thinking') and response.message.thinking:
            console.print(f"[dim]Thinking: {response.message.thinking[:200]}...[/dim]")
        
        console.print(f"[blue]{response.message.content}[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error generating answer: {e}[/red]")


def check_ollama_status() -> bool:
    """Check if Ollama is running and models are available."""
    console = Console()
    
    try:
        # Check if Ollama is running
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        response.raise_for_status()
        
        available_models = [model['name'] for model in response.json().get('models', [])]
        console.print(f"[green]✓ Ollama is running. Available models: {len(available_models)}[/green]")
        
        # Check if our embedding model is available
        if EMBEDDING_MODEL not in available_models:
            console.print(f"[yellow]Warning: {EMBEDDING_MODEL} not found. You may need to pull it:[/yellow]")
            console.print(f"[yellow]  ollama pull {EMBEDDING_MODEL}[/yellow]")
            return False
        
        console.print(f"[green]✓ {EMBEDDING_MODEL} is available[/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Ollama not available: {e}[/red]")
        console.print("[red]Please start Ollama and ensure models are pulled:[/red]")
        console.print(f"[red]  ollama pull {EMBEDDING_MODEL}[/red]")
        console.print("[red]  ollama pull qwen3:4b[/red]")
        return False


def main() -> None:
    """Main demo function."""
    console = Console()
    
    console.print("[bold blue]🔍 Ollama-based Grimm Tales RAG Search Demo[/bold blue]")
    console.print("=" * 50)
    
    # Check Ollama status
    if not check_ollama_status():
        return
    
    # Load document
    try:
        with open("demos/Kinder-und-Hausmärchen-der-Gebrüder-Grimm.txt", "r", encoding="utf8") as f:
            text = f.read()
        console.print(f"[green]✓ Loaded document: {len(text):,} characters[/green]")
    except FileNotFoundError:
        console.print("[red]✗ Error: 'demos/Kinder-und-Hausmärchen-der-Gebrüder-Grimm.txt' not found![/red]")
        return
    
    # Create chunks
    chunks = chunk_text(text)
    console.print(f"[green]✓ Created {len(chunks)} chunks[/green]")
    
    # Generate embeddings
    start_time = time.time()
    chunk_embeddings = generate_all_embeddings(chunks)
    embed_time = time.time() - start_time
    
    # Count successful embeddings
    successful_embeddings = sum(1 for emb in chunk_embeddings if emb is not None)
    console.print(f"[green]✓ Generated {successful_embeddings}/{len(chunks)} embeddings in {embed_time:.1f}s[/green]")
    
    if successful_embeddings == 0:
        console.print("[red]✗ No embeddings were generated successfully![/red]")
        return
    
    # Demo queries
    queries = [
        "What did the frog king promise the princess in exchange for her golden ball?",
        "What happened to Hansel and Gretel in the forest?",
        "What did Little Red Riding Hood's mother tell her to do?"
    ]
    
    console.print("\n" + "=" * 60)
    console.print("[bold]SEARCH DEMO[/bold]")
    console.print("=" * 60)
    
    for query in queries:
        start_time = time.time()
        results = search_chunks(query, chunks, chunk_embeddings)
        search_time = time.time() - start_time
        
        if results:
            display_results(query, results)
            console.print(f"[dim]Search completed in {search_time*1000:.0f}ms[/dim]")
            
            # Generate complete RAG answer using qwen3
            generate_answer(query, results)
        else:
            console.print(f"[red]No results found for query: {query}[/red]")
        
        console.print("-" * 50)
    
    console.print("[bold green]✓ Demo completed![/bold green]")


if __name__ == "__main__":
    main()
