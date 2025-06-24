
from typing import List, Dict, Tuple, Optional
import numpy as np
import requests
from dataclasses import dataclass
import re
from pathlib import Path
import chromadb
from chromadb.config import Settings
import json
import time

@dataclass
class Document:
    text: str
    metadata: Dict
    embedding: np.ndarray = None

class OllamaClient:
    def __init__(self, embedding_model: str = "nomic-embed-text:latest",
                 llm_model: str = "qwen3:4b"):
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.base_url = "http://localhost:11434/api"
        self.available_models = None
        
        # Fallback embedding models in order of preference
        self.embedding_fallbacks = [
            "nomic-embed-text:latest",
            "all-minilm:33m-l12-v2-fp16", 
            "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0",
            "all-minilm",
            "nomic-embed-text"
        ]
        
    def _get_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        if self.available_models is not None:
            return self.available_models
            
        try:
            response = requests.get(f"http://localhost:11434/api/tags")
            response.raise_for_status()
            models_data = response.json()
            self.available_models = [model['name'] for model in models_data.get('models', [])]
            return self.available_models
        except Exception as e:
            print(f"Warning: Could not fetch available models: {e}")
            return []
    
    def _find_working_embedding_model(self) -> Optional[str]:
        """Find a working embedding model from available options."""
        available_models = self._get_available_models()
        print(f"Available models: {available_models}")
        
        # Try each fallback model
        for model in self.embedding_fallbacks:
            if model in available_models:
                # Test if the model actually works for embeddings
                if self._test_embedding_model(model):
                    print(f"Using embedding model: {model}")
                    return model
                    
        print("Warning: No working embedding model found")
        return None
    
    def _test_embedding_model(self, model: str) -> bool:
        """Test if an embedding model works with a simple query."""
        try:
            data = {
                "model": model,
                "prompt": "test"
            }
            response = requests.post(f"{self.base_url}/embeddings", json=data, timeout=10)
            return response.status_code == 200 and 'embedding' in response.json()
        except:
            return False

    def get_embedding(self, text: str, max_retries: int = 3) -> np.ndarray:
        """Get embedding from Ollama API with fallback models and retry logic."""
        
        # Find working embedding model if not already set
        if not hasattr(self, '_working_embedding_model'):
            self._working_embedding_model = self._find_working_embedding_model()
            if not self._working_embedding_model:
                raise Exception("No working embedding model available")
        
        # Truncate text if too long (common issue with embeddings)
        if len(text) > 8000:
            text = text[:8000] + "..."
            
        data = {
            "model": self._working_embedding_model,
            "prompt": text
        }
        
        last_error = None
        for attempt in range(max_retries):
            try:
                response = requests.post(f"{self.base_url}/embeddings", json=data, timeout=30)
                response.raise_for_status()
                
                result = response.json()
                if 'embedding' not in result:
                    raise Exception(f"No embedding in response: {result}")
                    
                embedding = result["embedding"]
                return np.array(embedding)
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"Embedding attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(1)
                else:
                    # Try a different embedding model on final failure
                    available_models = self._get_available_models()
                    for fallback_model in self.embedding_fallbacks:
                        if (fallback_model != self._working_embedding_model and 
                            fallback_model in available_models):
                            print(f"Trying fallback embedding model: {fallback_model}")
                            data["model"] = fallback_model
                            try:
                                response = requests.post(f"{self.base_url}/embeddings", json=data, timeout=30)
                                response.raise_for_status()
                                result = response.json()
                                if 'embedding' in result:
                                    self._working_embedding_model = fallback_model
                                    return np.array(result["embedding"])
                            except:
                                continue
        
        raise Exception(f"Error getting embedding after {max_retries} attempts: {last_error}")

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
            response = requests.post(f"{self.base_url}/generate", json=data, timeout=60)
            response.raise_for_status()
            response_text = response.json()["response"]
            return response_text
        except Exception as e:
            raise Exception(f"Error generating response: {e}")


class SimpleChunker:
    def __init__(self, overlap_size: int = 50, max_chunk_size: int = 1000):
        self.overlap_size = overlap_size
        self.max_chunk_size = max_chunk_size

    def chunk_text(self, text: str) -> List[Document]:
        # Split text into paragraphs (assuming double newline as separator)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        chunks = []
        for i, para in enumerate(paragraphs):
            # Handle very long paragraphs by splitting them
            if len(para) > self.max_chunk_size:
                # Split long paragraph into sentences
                sentences = re.split(r'[.!?]+', para)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < self.max_chunk_size:
                        current_chunk += sentence.strip() + ". "
                    else:
                        if current_chunk:
                            chunks.append(Document(
                                text=current_chunk.strip(),
                                metadata={
                                    "chunk_id": len(chunks),
                                    "start_para": i,
                                    "text_preview": current_chunk[:100] + "...",
                                    "is_split_paragraph": True,
                                    "original_paragraphs": paragraphs
                                },
                                embedding=None
                            ))
                        current_chunk = sentence.strip() + ". "
                
                if current_chunk:
                    chunks.append(Document(
                        text=current_chunk.strip(),
                        metadata={
                            "chunk_id": len(chunks),
                            "start_para": i,
                            "text_preview": current_chunk[:100] + "...",
                            "is_split_paragraph": True,
                            "original_paragraphs": paragraphs
                        },
                        embedding=None
                    ))
            else:
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
                        "chunk_id": len(chunks),
                        "start_para": i,
                        "text_preview": chunk_text[:100] + "...",
                        "is_split_paragraph": False,
                        "original_paragraphs": paragraphs
                    },
                    embedding=None
                ))

        return chunks


class ChromaDBRetriever:
    def __init__(self, persist_directory: str = ".chromadb"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = "alice_documents"
        self.collection = None
        self.documents = []
        self.original_paragraphs = None
        
    def _setup_collection(self):
        """Setup or get existing ChromaDB collection"""
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"Using existing ChromaDB collection: {self.collection_name}")
            return True
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new ChromaDB collection: {self.collection_name}")
            return False

    def has_existing_data(self) -> bool:
        """Check if ChromaDB already has documents for this collection"""
        collection_exists = self._setup_collection()
        if collection_exists:
            try:
                existing_count = self.collection.count()
                return existing_count > 0
            except Exception as e:
                print(f"Error checking existing data: {e}")
                return False
        return False
    
    def load_existing_documents(self) -> List[Document]:
        """Load existing documents from ChromaDB"""
        try:
            existing_count = self.collection.count()
            print(f"Found {existing_count} existing documents in ChromaDB")
            
            # Get all documents from ChromaDB
            results = self.collection.get(include=['embeddings', 'metadatas', 'documents'])
            
            # Convert back to Document objects
            self.documents = []
            for i, (doc_id, text, metadata, embedding) in enumerate(zip(
                results['ids'], results['documents'], results['metadatas'], results['embeddings']
            )):
                # Restore original_paragraphs from metadata if available
                if 'original_paragraphs_json' in metadata:
                    metadata['original_paragraphs'] = json.loads(metadata['original_paragraphs_json'])
                    if self.original_paragraphs is None:
                        self.original_paragraphs = metadata['original_paragraphs']
                
                doc = Document(
                    text=text,
                    metadata=metadata,
                    embedding=np.array(embedding) if embedding else None
                )
                self.documents.append(doc)
            
            print(f"Loaded {len(self.documents)} documents from existing ChromaDB")
            return self.documents
            
        except Exception as e:
            print(f"Error loading existing documents: {e}")
            return []

    def add_documents(self, documents: List[Document]):
        """Add documents to ChromaDB collection"""
        self._setup_collection()
        
        # Store original paragraphs for context enhancement
        if documents and 'original_paragraphs' in documents[0].metadata:
            self.original_paragraphs = documents[0].metadata['original_paragraphs']
        
        # Add new documents to ChromaDB
        valid_documents = [doc for doc in documents if doc.embedding is not None]
        if not valid_documents:
            print("No documents with valid embeddings to add")
            return
            
        # Prepare data for ChromaDB
        ids = [f"doc_{i}" for i in range(len(valid_documents))]
        embeddings = [doc.embedding.tolist() for doc in valid_documents]
        metadatas = []
        texts = [doc.text for doc in valid_documents]
        
        # Prepare metadata (ChromaDB doesn't support complex objects, so serialize original_paragraphs)
        for doc in valid_documents:
            metadata = doc.metadata.copy()
            if 'original_paragraphs' in metadata:
                metadata['original_paragraphs_json'] = json.dumps(metadata['original_paragraphs'])
                del metadata['original_paragraphs']  # Remove the list from metadata
            metadatas.append(metadata)
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            self.documents = valid_documents
            print(f"Added {len(valid_documents)} documents to ChromaDB")
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")
            # Fallback to in-memory storage
            self.documents = valid_documents

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_relevant_chunks(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[Document, float]]:
        try:
            # Query ChromaDB for similar documents
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=['embeddings', 'metadatas', 'documents', 'distances']
            )
            
            # Convert results back to Document objects with similarity scores
            relevant_chunks = []
            for i, (doc_id, text, metadata, embedding, distance) in enumerate(zip(
                results['ids'][0], results['documents'][0], results['metadatas'][0], 
                results['embeddings'][0], results['distances'][0]
            )):
                # Calculate cosine similarity manually to ensure correctness
                doc_embedding = np.array(embedding)
                similarity = self.cosine_similarity(query_embedding, doc_embedding)
                
                # Restore original_paragraphs from JSON if available
                if 'original_paragraphs_json' in metadata:
                    metadata['original_paragraphs'] = json.loads(metadata['original_paragraphs_json'])
                elif self.original_paragraphs:
                    metadata['original_paragraphs'] = self.original_paragraphs
                
                doc = Document(
                    text=text,
                    metadata=metadata,
                    embedding=np.array(embedding)
                )
                relevant_chunks.append((doc, similarity))
            
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")
            # Fallback to manual similarity calculation
            similarities = []
            for doc in self.documents:
                if doc.embedding is not None:
                    similarity = self.cosine_similarity(query_embedding, doc.embedding)
                    similarities.append((doc, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            relevant_chunks = similarities[:top_k]

        # Enhance chunks with surrounding context
        enhanced_chunks = []
        paragraphs = self.original_paragraphs or (relevant_chunks[0][0].metadata.get('original_paragraphs', []) if relevant_chunks else [])

        for doc, score in relevant_chunks:
            chunk_id = doc.metadata.get('chunk_id', 0)

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
    def __init__(self, file_path: str, persist_directory: str = ".chromadb"):
        self.chunker = SimpleChunker()
        self.ollama_client = OllamaClient()
        self.retriever = ChromaDBRetriever(persist_directory)
        self.file_path = Path(file_path)
        
        # Check if we already have existing data in ChromaDB
        if self.retriever.has_existing_data():
            print("Using existing ChromaDB data...")
            self.documents = self.retriever.load_existing_documents()
        else:
            print("Creating new ChromaDB data...")
            # Read and process the text file
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
            failed_embeddings = 0
            
            for i, chunk in enumerate(chunks):
                try:
                    chunk.embedding = self.ollama_client.get_embedding(chunk.text)
                    chunk.metadata['embedding_dim'] = len(chunk.embedding)
                    successful_embeddings += 1
                    if (i + 1) % 10 == 0:
                        print(f"Processed {i + 1}/{len(chunks)} chunks (Success: {successful_embeddings}, Failed: {failed_embeddings})")
                except Exception as e:
                    print(f"Error embedding chunk {i}: {e}")
                    chunk.embedding = None
                    failed_embeddings += 1
                    
                    # If too many failures, stop and report
                    if failed_embeddings > 5 and successful_embeddings == 0:
                        raise Exception(f"Too many embedding failures ({failed_embeddings}). Check Ollama setup.")

            print(f"Embedding complete: {successful_embeddings} successful, {failed_embeddings} failed")
            
            if successful_embeddings == 0:
                raise Exception("No embeddings were successfully generated. Check Ollama setup and available models.")

            return chunks

        except Exception as e:
            raise Exception(f"Error processing file {self.file_path}: {str(e)}")

    def _generate_llm_answer(self, question: str, relevant_chunks: List[Tuple[Document, float]]) -> str:
        # Prepare context from relevant chunks
        context_parts = []
        for i, (doc, score) in enumerate(relevant_chunks, 1):
            context_parts.append(f"Passage {i} (Relevance: {score:.2f}):\n{doc.text}")

        context = "\n\n" + "\n\n".join(context_parts)

        # Create prompt for the LLM
        prompt = f"""Please answer the question based on the context provided below. Use only the context provided.
If you cannot answer based on the context, please say so.

Question: {question}

Context:{context}

Answer:"""

        try:
            return self.ollama_client.generate_response(prompt)
        except Exception as e:
            return f"Error generating LLM response: {str(e)}"

    def query(self, question: str) -> Tuple[List[Tuple[Document, float]], str, str]:
        if not question.strip():
            return [], "Please provide a question.", ""

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
                    f"[Paragraphs {doc.metadata.get('context_start', 'N/A')}-{doc.metadata.get('context_end', 'N/A')}]\n"
                    f"{doc.text}\n\n"
                )

            return relevant_chunks_with_scores, "".join(response_parts), llm_answer

        except Exception as e:
            return [], f"Error processing query: {str(e)}", ""


# Example usage
if __name__ == "__main__":
    try:
        print("Initializing Fixed RAG system with ChromaDB...")
        print("Make sure Ollama is running locally on port 11434")
        
        # Initialize RAG system
        rag = SimpleRAG("Alice_in_Wonderland.txt", persist_directory=".chromadb")

        print(f"\nProcessed {len(rag.documents)} chunks")
        if rag.retriever.documents:
            valid_docs = [doc for doc in rag.retriever.documents if doc.embedding is not None]
            print(f"Documents with valid embeddings: {len(valid_docs)}")
            if valid_docs:
                sample_doc = valid_docs[0]
                print(f"Embedding dimension: {len(sample_doc.embedding)}")

        # Example queries
        test_queries = [
            "What was Alice doing at the beginning of the story?",
            "What was written on the bottle that made Alice shrink?",
            "Where was the Cheshire Cat?",
            "What happens when Alice falls down the rabbit hole?",
        ]

        # Test each query
        for query in test_queries:
            print(f"\n{'=' * 50}\nQUERY: {query}\n{'=' * 50}")
            chunks_with_scores, detailed_response, llm_answer = rag.query(query)

            print("\nLLM's Answer:")
            print("-" * 25)
            print(llm_answer)
            print("\n")

            print("Retrieved passages:")
            for doc, score in chunks_with_scores:
                context_start = doc.metadata.get('context_start', 'N/A')
                context_end = doc.metadata.get('context_end', 'N/A')
                print(f"Passage (similarity: {score:.4f}): [Paragraphs {context_start}-{context_end}]")

    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Check available models: ollama list") 
        print("3. Pull embedding model if needed: ollama pull nomic-embed-text")
        print("4. Check Ollama API: curl http://localhost:11434/api/tags")
