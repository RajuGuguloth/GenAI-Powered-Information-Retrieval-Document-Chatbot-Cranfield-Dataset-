"""
RAG Index Building and Retrieval
Provides both LSA-based retrieval (reusing existing system) and FAISS-based retrieval
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import pickle
from pathlib import Path

# Import existing system components
from informationRetrieval import InformationRetrieval
from rag_utils import Document, RetrievalResult, DocumentProcessor, TextChunker

# Optional imports for advanced retrieval
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available. Using LSA-based retrieval only.")

try:
    from langchain.schema import BaseRetriever, Document as LangChainDocument
    from langchain.schema.retriever import BaseRetriever
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available. Using custom retriever interface.")

class LSABasedRetriever:
    """LSA-based retriever that reuses existing InformationRetrieval system"""
    
    def __init__(self, documents: List[Document], segmenter="punkt", tokenizer="ptb"):
        self.documents = documents
        self.doc_processor = DocumentProcessor(segmenter, tokenizer)
        self.ir_system = InformationRetrieval()
        self.doc_ids = []
        self.processed_docs = []
        self._build_index()
    
    def _build_index(self):
        """Build index using existing InformationRetrieval system"""
        # Preprocess documents
        self.processed_docs = self.doc_processor.preprocess_documents(self.documents)
        
        # Extract document IDs and processed content
        self.doc_ids = [doc.id for doc in self.documents]
        processed_content = [doc.processed_content for doc in self.processed_docs]
        
        # Build index using existing system
        self.ir_system.buildIndex(processed_content, self.doc_ids)
        print(f"Built LSA-based index for {len(self.documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve top-k documents using LSA-based retrieval"""
        # Preprocess query
        processed_query = self.doc_processor.preprocess_document(query)
        
        # Get ranked document IDs
        ranked_docs = self.ir_system.rank([processed_query])[0]
        
        # Create retrieval results
        results = []
        for i, doc_id in enumerate(ranked_docs[:top_k]):
            # Find document content
            doc = next((d for d in self.documents if d.id == str(doc_id)), None)
            if doc:
                # Calculate similarity score (inverse of rank)
                score = 1.0 / (i + 1)
                result = RetrievalResult(
                    document_id=str(doc_id),
                    content=doc.content,
                    score=score,
                    metadata=doc.metadata
                )
                results.append(result)
        
        return results

class FAISSRetriever:
    """FAISS-based retriever using sentence transformers"""
    
    def __init__(self, documents: List[Document], model_name: str = "all-MiniLM-L6-v2"):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS and sentence-transformers are required for this retriever")
        
        self.documents = documents
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.doc_embeddings = []
        self._build_index()
    
    def _build_index(self):
        """Build FAISS index from document embeddings"""
        # Generate embeddings for all documents
        doc_texts = [doc.content for doc in self.documents]
        self.doc_embeddings = self.model.encode(doc_texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.doc_embeddings)
        self.index.add(self.doc_embeddings.astype('float32'))
        
        print(f"Built FAISS index for {len(self.documents)} documents with dimension {dimension}")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve top-k documents using FAISS similarity search"""
        # Generate query embedding
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search index
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Create retrieval results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                doc = self.documents[idx]
                result = RetrievalResult(
                    document_id=doc.id,
                    content=doc.content,
                    score=float(score),
                    metadata=doc.metadata
                )
                results.append(result)
        
        return results

class HybridRetriever:
    """Hybrid retriever that combines LSA and FAISS results"""
    
    def __init__(self, documents: List[Document], use_lsa: bool = True, use_faiss: bool = False):
        self.documents = documents
        self.retrievers = []
        
        if use_lsa:
            self.lsa_retriever = LSABasedRetriever(documents)
            self.retrievers.append(("lsa", self.lsa_retriever))
        
        if use_faiss and FAISS_AVAILABLE:
            self.faiss_retriever = FAISSRetriever(documents)
            self.retrievers.append(("faiss", self.faiss_retriever))
        
        if not self.retrievers:
            raise ValueError("At least one retriever must be enabled")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve documents using hybrid approach"""
        all_results = {}
        
        # Get results from each retriever
        for retriever_name, retriever in self.retrievers:
            try:
                results = retriever.retrieve(query, top_k)
                for result in results:
                    if result.document_id not in all_results:
                        all_results[result.document_id] = result
                    else:
                        # Combine scores if document appears in multiple retrievers
                        all_results[result.document_id].score = max(
                            all_results[result.document_id].score, 
                            result.score
                        )
            except Exception as e:
                print(f"Error in {retriever_name} retriever: {e}")
        
        # Sort by score and return top-k
        sorted_results = sorted(all_results.values(), key=lambda x: x.score, reverse=True)
        return sorted_results[:top_k]

class LangChainRetrieverWrapper:
    """Wrapper to make our retrievers compatible with LangChain"""
    
    def __init__(self, retriever, top_k: int = 5):
        self.retriever = retriever
        self.top_k = top_k
    
    def get_relevant_documents(self, query: str) -> List[LangChainDocument]:
        """Get relevant documents for LangChain compatibility"""
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for this wrapper")
        
        results = self.retriever.retrieve(query, self.top_k)
        
        # Convert to LangChain documents
        langchain_docs = []
        for result in results:
            doc = LangChainDocument(
                page_content=result.content,
                metadata={
                    "document_id": result.document_id,
                    "score": result.score,
                    **result.metadata
                }
            )
            langchain_docs.append(doc)
        
        return langchain_docs
    
    def invoke(self, query: str) -> List[LangChainDocument]:
        """Alternative method for LangChain compatibility"""
        return self.get_relevant_documents(query)

class IndexBuilder:
    """Main class for building and managing document indices"""
    
    def __init__(self, dataset_path: str = "cranfield/", chunk_size: int = 500):
        self.dataset_path = dataset_path
        self.chunk_size = chunk_size
        self.documents = []
        self.chunked_documents = []
        self.retriever = None
    
    def load_andProcessDocuments(self, chunk_documents: bool = True):
        """Load documents and optionally chunk them"""
        from rag_utils import load_cranfield_dataset
        
        # Load documents from Cranfield dataset
        self.documents, _, _ = load_cranfield_dataset(self.dataset_path)
        
        if not self.documents:
            print("No documents loaded. Checking for text files in common directories...")
            self._loadFromTextFiles()
        
        if chunk_documents and self.documents:
            # Chunk documents for better retrieval
            chunker = TextChunker(chunk_size=self.chunk_size)
            self.chunked_documents = chunker.chunk_documents(self.documents)
            print(f"Created {len(self.chunked_documents)} chunks from {len(self.documents)} documents")
        else:
            self.chunked_documents = self.documents
        
        return self.chunked_documents
    
    def _loadFromTextFiles(self):
        """Load documents from text files in common directories"""
        common_dirs = ["docs/", "datasets/", "data/", "texts/"]
        
        for dir_name in common_dirs:
            if os.path.exists(dir_name):
                print(f"Searching for text files in {dir_name}")
                for file_path in Path(dir_name).rglob("*.txt"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        doc = Document(
                            id=str(file_path),
                            content=content,
                            metadata={"source": str(file_path), "type": "text_file"}
                        )
                        self.documents.append(doc)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
        
        print(f"Loaded {len(self.documents)} documents from text files")
    
    def buildRetriever(self, retriever_type: str = "auto", **kwargs):
        """Build a retriever of the specified type"""
        if not self.chunked_documents:
            raise ValueError("No documents loaded. Call loadAndProcessDocuments() first.")
        
        if retriever_type == "auto":
            # Try to detect existing retriever function
            retriever_type = self._detectRetrieverType()
        
        if retriever_type == "lsa":
            self.retriever = LSABasedRetriever(self.chunked_documents, **kwargs)
        elif retriever_type == "faiss":
            if not FAISS_AVAILABLE:
                print("FAISS not available, falling back to LSA")
                self.retriever = LSABasedRetriever(self.chunked_documents, **kwargs)
            else:
                self.retriever = FAISSRetriever(self.chunked_documents, **kwargs)
        elif retriever_type == "hybrid":
            self.retriever = HybridRetriever(self.chunked_documents, **kwargs)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
        
        return self.retriever
    
    def _detectRetrieverType(self) -> str:
        """Detect the best retriever type based on available components"""
        # Check if we have the existing InformationRetrieval system
        if hasattr(self, 'documents') and self.documents:
            return "lsa"  # Default to LSA since it's already implemented
        
        # Check if FAISS is available
        if FAISS_AVAILABLE:
            return "faiss"
        
        # Fallback to LSA
        return "lsa"
    
    def saveIndex(self, filepath: str):
        """Save the built index to disk"""
        if not self.retriever:
            raise ValueError("No retriever built. Call buildRetriever() first.")
        
        # Save index data
        index_data = {
            "retriever_type": type(self.retriever).__name__,
            "document_count": len(self.chunked_documents),
            "chunk_size": self.chunk_size
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"Index metadata saved to {filepath}")
    
    def loadIndex(self, filepath: str):
        """Load index from disk"""
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)
        
        print(f"Loaded index: {index_data}")
        return index_data

def createRetriever(documents: List[Document], retriever_type: str = "auto", **kwargs):
    """Convenience function to create a retriever"""
    builder = IndexBuilder()
    builder.documents = documents
    builder.chunked_documents = documents
    return builder.buildRetriever(retriever_type, **kwargs)

def detectExistingRetriever() -> Optional[str]:
    """Detect if there's an existing retriever function in the codebase"""
    # This function would analyze the existing code to find retriever functions
    # For now, we'll assume the InformationRetrieval class exists
    try:
        from informationRetrieval import InformationRetrieval
        return "lsa"
    except ImportError:
        return None

if __name__ == "__main__":
    # Example usage
    builder = IndexBuilder()
    documents = builder.loadAndProcessDocuments()
    
    if documents:
        retriever = builder.buildRetriever("auto")
        print(f"Built retriever: {type(retriever).__name__}")
        
        # Test retrieval
        test_query = "aerodynamics and fluid dynamics"
        results = retriever.retrieve(test_query, top_k=3)
        
        print(f"\nRetrieval results for: '{test_query}'")
        for i, result in enumerate(results):
            print(f"{i+1}. Doc {result.document_id} (Score: {result.score:.3f})")
            print(f"   Content: {result.content[:100]}...")
            print()
    else:
        print("No documents found to build index")
