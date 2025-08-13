"""
RAG Utilities for Information Retrieval System
Provides document processing, retrieval, and LLM integration capabilities
"""

import json
import os
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from pathlib import Path

# Import existing utilities
from util import *
from informationRetrieval import InformationRetrieval

# Minimal regex-based sentence segmenter fallback
import re as _re

class SentenceSegmentation:
    """Minimal regex-based sentence segmenter used as a safe fallback.

    Provides naive() and punkt() methods to match the existing interface.
    """

    def naive(self, text: str) -> List[str]:
        if not text:
            return []
        parts = _re.split(r"(?<=[\.\?\!])\s+", text)
        return [p.strip() for p in parts if p and p.strip()]

    def punkt(self, text: str) -> List[str]:
        # For this minimal implementation, use the same logic as naive
        return self.naive(text)

@dataclass
class Document:
    """Document representation for RAG system"""
    id: str
    content: str
    processed_content: List[List[str]] = None
    metadata: Dict[str, Any] = None

@dataclass
class RetrievalResult:
    """Result from document retrieval"""
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = None

class DocumentProcessor:
    """Processes documents for RAG system using existing preprocessing pipeline"""
    
    def __init__(self, segmenter="punkt", tokenizer="ptb"):
        self.segmenter = segmenter
        self.tokenizer = tokenizer

        # Try to use the project's preprocessing classes; otherwise, fall back
        try:
            from sentenceSegmentation import SentenceSegmentation as _ExtSentenceSegmentation
            from tokenization import Tokenization
            from inflectionReduction import InflectionReduction
            from stopwordRemoval import StopwordRemoval

            self.sentenceSegmenter = _ExtSentenceSegmentation()
            self.tokenizer_obj = Tokenization()
            self.inflectionReducer = InflectionReduction()
            self.stopwordRemover = StopwordRemoval()
        except Exception as e:
            print(f"Warning: preprocessing classes not fully available: {e}")
            # Fallbacks
            self.sentenceSegmenter = SentenceSegmentation()
            self.tokenizer_obj = None
            self.inflectionReducer = None
            self.stopwordRemover = None
    
    def segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences using available segmenter or fallback."""
        if not self.sentenceSegmenter:
            return [s.strip() for s in _re.split(r"(?<=[\.\?\!])\s+", text or "") if s.strip()]
        if self.segmenter == "naive":
            return self.sentenceSegmenter.naive(text)
        elif self.segmenter == "punkt":
            return self.sentenceSegmenter.punkt(text)
        return self.sentenceSegmenter.punkt(text)
    
    def tokenize(self, sentences: List[str]) -> List[List[str]]:
        """Tokenize sentences using available tokenizer or simple split fallback."""
        if not self.tokenizer_obj:
            return [(s or "").split() for s in sentences]
        tokenized = []
        for sentence in sentences:
            if self.tokenizer == "naive":
                tokens = self.tokenizer_obj.naive(sentence)
            elif self.tokenizer == "ptb":
                tokens = self.tokenizer_obj.pennTreeBank(sentence)
            else:
                tokens = self.tokenizer_obj.pennTreeBank(sentence)
            tokenized.append(tokens)
        return tokenized
    
    def reduce_inflection(self, tokenized_sentences: List[List[str]]) -> List[List[str]]:
        """Reduce inflection using stemmer/lemmatizer if available; else pass-through."""
        if not self.inflectionReducer:
            return tokenized_sentences
        reduced = []
        for sentence in tokenized_sentences:
            reduced_sentence = self.inflectionReducer.reduce(sentence)
            reduced.append(reduced_sentence)
        return reduced
    
    def remove_stopwords(self, processed_sentences: List[List[str]]) -> List[List[str]]:
        """Remove stopwords if remover available; else pass-through."""
        if not self.stopwordRemover:
            return processed_sentences
        cleaned = []
        for sentence in processed_sentences:
            cleaned_sentence = self.stopwordRemover.fromList(sentence)
            cleaned.append(cleaned_sentence)
        return cleaned
    
    def preprocess_document(self, text: str) -> List[List[str]]:
        """Complete preprocessing pipeline for a single document"""
        segmented = self.segment_sentences(text)
        tokenized = self.tokenize(segmented)
        reduced = self.reduce_inflection(tokenized)
        cleaned = self.remove_stopwords(reduced)
        return cleaned
    
    def preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """Preprocess all documents"""
        for doc in documents:
            doc.processed_content = self.preprocess_document(doc.content)
        return documents

class CranfieldDatasetLoader:
    """Loads and processes Cranfield dataset for RAG system"""
    
    def __init__(self, dataset_path: str = "cranfield/"):
        self.dataset_path = dataset_path
        self.documents = []
        self.queries = []
        self.qrels = []
    
    def load_documents(self) -> List[Document]:
        """Load documents from Cranfield dataset"""
        try:
            docs_file = os.path.join(self.dataset_path, "cran_docs.json")
            if os.path.exists(docs_file):
                with open(docs_file, 'r') as f:
                    docs_json = json.load(f)
                
                self.documents = []
                for doc in docs_json:
                    document = Document(
                        id=str(doc["id"]),
                        content=doc["body"],
                        metadata={"title": doc.get("title", ""), "author": doc.get("author", "")}
                    )
                    self.documents.append(document)
                
                print(f"Loaded {len(self.documents)} documents from Cranfield dataset")
                return self.documents
            else:
                print(f"Documents file not found at {docs_file}")
                return []
        except Exception as e:
            print(f"Error loading documents: {e}")
            return []
    
    def load_queries(self) -> List[Dict]:
        """Load queries from Cranfield dataset"""
        try:
            queries_file = os.path.join(self.dataset_path, "cran_queries.json")
            if os.path.exists(queries_file):
                with open(queries_file, 'r') as f:
                    self.queries = json.load(f)
                print(f"Loaded {len(self.queries)} queries from Cranfield dataset")
                return self.queries
            else:
                print(f"Queries file not found at {queries_file}")
                return []
        except Exception as e:
            print(f"Error loading queries: {e}")
            return []
    
    def load_qrels(self) -> List[Dict]:
        """Load relevance judgments from Cranfield dataset"""
        try:
            qrels_file = os.path.join(self.dataset_path, "cran_qrels.json")
            if os.path.exists(qrels_file):
                with open(qrels_file, 'r') as f:
                    self.qrels = json.load(f)
                print(f"Loaded {len(self.qrels)} relevance judgments from Cranfield dataset")
                return self.qrels
            else:
                print(f"Qrels file not found at {qrels_file}")
                return []
        except Exception as e:
            print(f"Error loading qrels: {e}")
            return []

class TextChunker:
    """Chunks documents into smaller pieces for better retrieval"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_document(self, document: Document) -> List[Document]:
        """Split a document into overlapping chunks"""
        if not document.content:
            return []
        
        # Simple sentence-based chunking
        sentences = document.content.split('. ')
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk.strip():
                    chunk_doc = Document(
                        id=f"{document.id}_chunk_{chunk_id}",
                        content=current_chunk.strip(),
                        metadata={
                            "original_id": document.id,
                            "chunk_id": chunk_id,
                            "chunk_type": "sentence_based",
                            **document.metadata
                        }
                    )
                    chunks.append(chunk_doc)
                    chunk_id += 1
                
                # Start new chunk with overlap
                current_chunk = sentence + ". "
        
        # Add the last chunk
        if current_chunk.strip():
            chunk_doc = Document(
                id=f"{document.id}_chunk_{chunk_id}",
                content=current_chunk.strip(),
                metadata={
                    "original_id": document.id,
                    "chunk_id": chunk_id,
                    "chunk_type": "sentence_based",
                    **document.metadata
                }
            )
            chunks.append(chunk_doc)
        
        return chunks
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Chunk all documents"""
        chunked_docs = []
        for doc in documents:
            chunks = self.chunk_document(doc)
            chunked_docs.extend(chunks)
        return chunked_docs

class QueryProcessor:
    """Processes user queries for RAG system"""
    
    def __init__(self, document_processor: DocumentProcessor):
        self.doc_processor = document_processor
    
    def preprocess_query(self, query: str) -> List[List[str]]:
        """Preprocess a single query using existing pipeline"""
        return self.doc_processor.preprocess_document(query)
    
    def extract_keywords(self, processed_query: List[List[str]]) -> List[str]:
        """Extract keywords from processed query"""
        keywords = []
        for sentence in processed_query:
            keywords.extend(sentence)
        return [kw for kw in keywords if kw.strip()]

class RAGEvaluator:
    """Evaluates RAG system performance"""
    
    def __init__(self):
        pass
    
    def evaluate_source_grounding(self, answer: str, retrieved_docs: List[RetrievalResult]) -> Dict[str, float]:
        """Evaluate how well the answer is grounded in retrieved documents"""
        if not retrieved_docs:
            return {"grounding_score": 0.0, "coverage": 0.0}
        
        # Simple keyword overlap scoring
        answer_words = set(answer.lower().split())
        doc_words = set()
        for doc in retrieved_docs:
            doc_words.update(doc.content.lower().split())
        
        overlap = len(answer_words.intersection(doc_words))
        grounding_score = overlap / len(answer_words) if answer_words else 0.0
        coverage = overlap / len(doc_words) if doc_words else 0.0
        
        return {
            "grounding_score": grounding_score,
            "coverage": coverage,
            "overlap_count": overlap,
            "answer_word_count": len(answer_words),
            "doc_word_count": len(doc_words)
        }
    
    def evaluate_factual_correctness(self, answer: str, retrieved_docs: List[RetrievalResult]) -> Dict[str, Any]:
        """Evaluate factual correctness of the answer"""
        # This is a simplified evaluation - in practice, you might use more sophisticated methods
        return {
            "has_citations": self._check_citations(answer),
            "answer_length": len(answer),
            "retrieved_doc_count": len(retrieved_docs)
        }
    
    def _check_citations(self, answer: str) -> bool:
        """Check if answer contains document citations"""
        # Look for patterns like [doc_id], (doc_id), or similar citation formats
        citation_patterns = [
            r'\[.*?\]',  # [doc_id]
            r'\(.*?\)',  # (doc_id)
            r'document\s+\d+',  # document 123
            r'doc\s+\d+',  # doc 123
        ]
        
        for pattern in citation_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                return True
        return False

def load_cranfield_dataset(dataset_path: str = "cranfield/") -> Tuple[List[Document], List[Dict], List[Dict]]:
    """Convenience function to load Cranfield dataset"""
    loader = CranfieldDatasetLoader(dataset_path)
    documents = loader.load_documents()
    queries = loader.load_queries()
    qrels = loader.load_qrels()
    return documents, queries, qrels

def create_document_from_text(text: str, doc_id: str = None) -> Document:
    """Create a Document object from raw text"""
    if doc_id is None:
        doc_id = f"doc_{hash(text) % 10000}"
    
    return Document(
        id=doc_id,
        content=text,
        metadata={"source": "text_input", "length": len(text)}
    )
