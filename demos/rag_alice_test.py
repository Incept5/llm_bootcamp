
from typing import List, Dict, Tuple
import numpy as np
import requests
from dataclasses import dataclass
import re
from pathlib import Path

@dataclass
class Document:
    text: str
    metadata: Dict
    embedding: np.ndarray = None

class OllamaClient:
    def __init__(self, embedding_model: str = "nomic-embed-text",
                 llm_model: str = "qwen3:4b"):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.base_url = "http://localhost:11434/api"

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Ollama API."""
        data = {
            "model": self.embedding_model,
            "prompt": text
        }
        try:
            response = requests.post(f"{self.base_url}/embeddings", json=data)
            response.raise_for_status()
            result = response.json()
            
            # Handle different response formats
            if "embedding" in result:
                embedding = result["embedding"]
            elif "embeddings" in result:
                embedding = result["embeddings"][0]
            else:
                raise Exception(f"Unexpected response format: {result}")
                
            return np.array(embedding)
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error getting embedding: {e}")
        except Exception as e:
            raise Exception(f"Error processing embedding response: {e}")

    def generate_response(self, prompt: str) -> str:
        """Generate text response from Ollama API."""
        print(f"\nCalling Ollama LLM ({self.llm_model})...")
        data = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 8192,
                "temperature": 0.3
            }
        }

        try:
            response = requests.post(f"{self.base_url}/generate", json=data)
            response.raise_for_status()
            response_text = response.json()["response"]
            return response_text
        except Exception as e:
            raise Exception(f"Error generating response: {e}")


class SimpleChunker:
    def __init__(self, overlap_size: int = 50, max_chunks: int = 20):
        self.overlap_size = overlap_size
        self.max_chunks = max_chunks

    def chunk_text(self, text: str) -> List[Document]:
        # Split text into paragraphs (assuming double newline as separator)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Limit the number of paragraphs for testing
        paragraphs = paragraphs[:self.max_chunks]
        print(f"Processing first {len(paragraphs)} paragraphs for testing")

        chunks = []
        for i, para in enumerate(paragraphs):
            # Create overlapping chunks by including part of the next paragraph
            if i < len(paragraphs) - 1:
                next_para = paragraphs[i + 1]
                overlap = next_para[:min(self.overlap_size, len(next_para))]
                chunk_text = para + "\n" + overlap
            else:
                chunk_text = para

            chunks.append(Document(
                text=chunk_text,
                metadata={
                    "chunk_id": i,
                    "start_para": i,
                    "text_preview": chunk_text[:100] + "...",
                    "total_chunks": len(paragraphs),
                    "original_paragraphs": paragraphs  # Store all paragraphs for context
                },
                embedding=None
            ))

        return chunks


class SimpleRetriever:
    def __init__(self):
        self.documents = []

    def add_documents(self, documents: List[Document]):
        self.documents = [doc for doc in documents if doc.embedding is not None]
        print(f"Added {len(self.documents)} documents with valid embeddings")

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_relevant_chunks(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[Document, float]]:
        if not self.documents:
            return []
            
        # Get initial similarity scores
        similarities = []
        for doc in self.documents:
            similarity = self.cosine_similarity(query_embedding, doc.embedding)
            similarities.append((doc, similarity))

        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_chunks = similarities[:top_k]

        # Enhance chunks with surrounding context
        enhanced_chunks = []
        paragraphs = self.documents[0].metadata['original_paragraphs']

        for doc, score in top_chunks:
            chunk_id = doc.metadata['chunk_id']

            # Get surrounding paragraphs
            start_idx = max(0, chunk_id - 1)
            end_idx = min(len(paragraphs), chunk_id + 2)

            # Combine paragraphs into enhanced chunk
            enhanced_text = "\n\n".join(paragraphs[start_idx:end_idx])

            # Create new document with enhanced text
            enhanced_doc = Document(
                text=enhanced_text,
                metadata={
                    **doc.metadata,
                    "original_chunk_id": chunk_id,
                    "context_start": start_idx,
                    "context_end": end_idx - 1
                },
                embedding=doc.embedding
            )

            enhanced_chunks.append((enhanced_doc, score))

        return enhanced_chunks


class SimpleRAG:
    def __init__(self, file_path: str, max_chunks: int = 20):
        self.chunker = SimpleChunker(max_chunks=max_chunks)
        self.ollama_client = OllamaClient()
        self.retriever = SimpleRetriever()

        # Read and process the text file
        self.file_path = Path(file_path)
        self.documents = self._load_and_process_text()
        self.retriever.add_documents(self.documents)

    def _load_and_process_text(self) -> List[Document]:
        try:
            if not self.file_path.exists():
                raise FileNotFoundError(f"File not found: {self.file_path}")

            with open(self.file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            # Basic text cleanup
            text = re.sub(r'\r\n', '\n', text)  # Normalize line endings
            text = re.sub(r'\n{3,}', '\n\n', text)  # Normalize paragraph spacing

            # Chunk the text
            print("Chunking text...")
            chunks = self.chunker.chunk_text(text)
            print(f"Created {len(chunks)} chunks")

            # Generate embeddings for all chunks
            print("Generating embeddings...")
            successful_embeddings = 0
            
            for i, chunk in enumerate(chunks):
                try:
                    print(f"Processing chunk {i+1}/{len(chunks)}...")
                    chunk.embedding = self.ollama_client.get_embedding(chunk.text)
                    chunk.metadata['embedding_dim'] = len(chunk.embedding)
                    successful_embeddings += 1
                except Exception as e:
                    print(f"Error embedding chunk {i}: {e}")
                    chunk.embedding = None

            print(f"Successfully generated {successful_embeddings}/{len(chunks)} embeddings")
            return chunks

        except Exception as e:
            raise Exception(f"Error processing file {self.file_path}: {str(e)}")

    def query(self, question: str) -> Tuple[List[Tuple[Document, float]], str]:
        if not question.strip():
            return [], "Please provide a question."

        if not self.retriever.documents:
            return [], "No documents with embeddings available for querying."

        try:
            # Generate embedding for the query
            print(f"Generating embedding for query: {question}")
            query_embedding = self.ollama_client.get_embedding(question)

            # Get relevant chunks with similarity scores
            relevant_chunks_with_scores = self.retriever.get_relevant_chunks(query_embedding)

            # Generate LLM answer
            if relevant_chunks_with_scores:
                context_parts = []
                for i, (doc, score) in enumerate(relevant_chunks_with_scores, 1):
                    context_parts.append(f"Passage {i} (Relevance: {score:.2f}):{doc.text}")

                context = "\n\n" + "\n".join(context_parts)

                # Create prompt for the LLM
                prompt = f"""Please answer the question based on the context, use only the context provided.
If you cannot answer based on the context, please say so.

Question: {question}

Context:{context}
"""
                llm_answer = self.ollama_client.generate_response(prompt)
            else:
                llm_answer = "No relevant passages found."

            return relevant_chunks_with_scores, llm_answer

        except Exception as e:
            return [], f"Error processing query: {str(e)}"


# Test script
if __name__ == "__main__":
    try:
        print("=== RAG SYSTEM TEST ===")
        print("Initializing RAG system with limited chunks for testing...")
        print("Make sure Ollama is running locally on port 11434")
        
        rag = SimpleRAG("demos/Alice_in_Wonderland.txt", max_chunks=20)

        print(f"\nProcessed {len(rag.documents)} chunks")
        
        # Check if we have any successful embeddings
        successful_docs = [doc for doc in rag.documents if doc.embedding is not None]
        if successful_docs:
            print(f"Successfully embedded {len(successful_docs)} chunks")
            print(f"Embedding dimension: {successful_docs[0].metadata['embedding_dim']}")
        else:
            print("No successful embeddings - cannot proceed with queries")
            exit(1)

        # Test a simple query
        test_query = "What was Alice doing at the beginning of the story?"
        print(f"\n{'=' * 50}")
        print(f"TEST QUERY: {test_query}")
        print('=' * 50)
        
        chunks_with_scores, llm_answer = rag.query(test_query)

        if chunks_with_scores:
            print("\nLLM's Answer:")
            print("=" * 20)
            print(llm_answer)
            print("\nRelevant passages:")
            for doc, score in chunks_with_scores:
                print(f"\nPassage (similarity: {score:.4f}):")
                print(f"Text preview: {doc.text[:200]}...")
        else:
            print("No relevant passages found or error occurred")

        print("\n=== TEST COMPLETED ===")

    except Exception as e:
        print(f"Error: {str(e)}")
