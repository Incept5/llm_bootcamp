
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
    def __init__(self, overlap_size: int = 50):
        self.overlap_size = overlap_size

    def chunk_text(self, text: str) -> List[Document]:
        # Split text into paragraphs (assuming double newline as separator)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

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
    def __init__(self, file_path: str):
        self.chunker = SimpleChunker()
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
                    chunk.embedding = self.ollama_client.get_embedding(chunk.text)
                    chunk.metadata['embedding_dim'] = len(chunk.embedding)
                    successful_embeddings += 1
                    if (i + 1) % 10 == 0:
                        print(f"Processed {i + 1}/{len(chunks)} chunks ({successful_embeddings} successful)")
                except Exception as e:
                    print(f"Error embedding chunk {i}: {e}")
                    chunk.embedding = None

            print(f"Successfully generated {successful_embeddings}/{len(chunks)} embeddings")
            return chunks

        except Exception as e:
            raise Exception(f"Error processing file {self.file_path}: {str(e)}")

    def _generate_llm_answer(self, question: str, relevant_chunks: List[Tuple[Document, float]]) -> str:
        # Prepare context from relevant chunks
        context_parts = []
        for i, (doc, score) in enumerate(relevant_chunks, 1):
            context_parts.append(f"Passage {i} (Relevance: {score:.2f}):{doc.text}")

        context = "\n\n" + "\n".join(context_parts)

        # Create prompt for the LLM
        prompt = f"""Please answer the question based on the context, use only the context provided.
If you cannot answer based on the context, please say so.

Question: {question}

Context:{context}
"""

        try:
            return self.ollama_client.generate_response(prompt)
        except Exception as e:
            return f"Error generating LLM response: {str(e)}"

    def query(self, question: str) -> Tuple[List[Tuple[Document, float]], str, str]:
        if not question.strip():
            return [], "Please provide a question.", ""

        if not self.retriever.documents:
            return [], "No documents with embeddings available for querying.", ""

        try:
            # Generate embedding for the query
            query_embedding = self.ollama_client.get_embedding(question)

            # Get relevant chunks with similarity scores
            relevant_chunks_with_scores = self.retriever.get_relevant_chunks(query_embedding)

            # Generate LLM answer
            llm_answer = self._generate_llm_answer(question, relevant_chunks_with_scores)

            # Create detailed response with similarity scores
            response_parts = [f"Query: {question}\n\n"]
            response_parts.append(f"Found {len(relevant_chunks_with_scores)} relevant passages:\n\n")

            for doc, score in relevant_chunks_with_scores:
                response_parts.append(
                    f"Passage (similarity: {score:.4f}):\n"
                    f"[Paragraphs {doc.metadata['context_start']}-{doc.metadata['context_end']}]\n"
                    f"{doc.text}\n\n"
                )

            return relevant_chunks_with_scores, "".join(response_parts), llm_answer

        except Exception as e:
            return [], f"Error processing query: {str(e)}", ""


# Example usage
if __name__ == "__main__":
    try:
        print("Initializing RAG system...")
        print("Make sure Ollama is running locally on port 11434")
        print("Required models: nomic-embed-text, qwen3:4b")
        print("Install with: ollama pull nomic-embed-text && ollama pull qwen3:4b")
        
        rag = SimpleRAG("../Alice in Wonderland.txt")

        print(f"\nProcessed {len(rag.documents)} chunks")
        
        # Check if we have any successful embeddings
        successful_docs = [doc for doc in rag.documents if doc.embedding is not None]
        if successful_docs:
            print(f"Successfully embedded {len(successful_docs)} chunks")
            print(f"Embedding dimension: {successful_docs[0].metadata['embedding_dim']}")
        else:
            print("No successful embeddings - cannot proceed with queries")
            print("Please check:")
            print("1. Ollama is running: ollama serve")
            print("2. Embedding model is installed: ollama pull nomic-embed-text")
            print("3. LLM model is installed: ollama pull qwen3:4b")
            exit(1)

        # Example queries
        test_queries = [
            "What was Alice doing at the beginning of the story?",
            "What was written on the bottle that made Alice shrink?",
            "Where was the Cheshire Cat?",
            "Is there mention of an Essex cat?",
            "What happens when Alice falls down the rabbit hole?",
            "When Alice was talking to the Duchess, describe how the queen stood when she surprised Alice",
        ]

        # Test each query
        for query in test_queries:
            print(f"\n{'=' * 50}\nQUERY: {query}\n{'=' * 50}")
            chunks_with_scores, detailed_response, llm_answer = rag.query(query)

            if chunks_with_scores:
                print("\nLLM's Interpreted Answer:")
                print("-------------------------")
                print(llm_answer)
                print("\n")

                print("Retrieved passages with surrounding context:")
                for doc, score in chunks_with_scores:
                    print(f"Passage (similarity: {score:.4f}): [Paragraphs {doc.metadata['context_start']}-{doc.metadata['context_end']}]")
            else:
                print("No relevant passages found or error occurred")

    except Exception as e:
        print(f"Error: {str(e)}")
