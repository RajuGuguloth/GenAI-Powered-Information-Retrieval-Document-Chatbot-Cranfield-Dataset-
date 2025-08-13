"""
RAG Server with Gradio Chat Interface
Main server that integrates retrieval, generation, and chat interface
"""

import os
import json
import time
from typing import List, Dict, Tuple, Optional, Any
import gradio as gr
from pathlib import Path

# Import RAG components
from rag_utils import Document, RetrievalResult, RAGEvaluator
from rag_build_index import IndexBuilder, LSABasedRetriever, FAISSRetriever, HybridRetriever

# Optional imports for LLM integration
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available. Using mock LLM responses.")

try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("HuggingFace transformers not available. Using mock LLM responses.")

class RAGSystem:
    """Main RAG system that combines retrieval and generation"""
    
    def __init__(self, 
                 dataset_path: str = "cranfield/",
                 retriever_type: str = "auto",
                 llm_provider: str = "openai",
                 top_k: int = 5,
                 chunk_size: int = 500):
        
        self.dataset_path = dataset_path
        self.retriever_type = retriever_type
        self.llm_provider = llm_provider
        self.top_k = top_k
        self.chunk_size = chunk_size
        
        # Initialize components
        self.retriever = None
        self.llm = None
        self.evaluator = RAGEvaluator()
        self.documents = []
        
        # Build the system
        self._buildSystem()
    
    def _buildSystem(self):
        """Build the RAG system components"""
        print("Building RAG system...")
        
        # Build document index and retriever
        builder = IndexBuilder(self.dataset_path, self.chunk_size)
        self.documents = builder.load_andProcessDocuments(chunk_documents=True)
        
        if not self.documents:
            print("Warning: No documents loaded. System may not function properly.")
            return
        
        self.retriever = builder.buildRetriever(self.retriever_type)
        print(f"Built retriever: {type(self.retriever).__name__}")
        
        # Initialize LLM
        self._initializeLLM()
    
    def _initializeLLM(self):
        """Initialize the language model"""
        if self.llm_provider == "openai" and OPENAI_AVAILABLE:
            # Check for API key
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                openai.api_key = api_key
                self.llm = "openai"
                print("OpenAI LLM initialized")
            else:
                print("OpenAI API key not found. Using mock LLM.")
                self.llm = "mock"
        
        elif self.llm_provider == "huggingface" and HUGGINGFACE_AVAILABLE:
            try:
                # Use a small, fast model for local inference
                model_name = "gpt2"  # You can change this to other models
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                
                # Add padding token if not present
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.llm = "huggingface"
                print(f"HuggingFace LLM initialized with {model_name}")
            except Exception as e:
                print(f"Error initializing HuggingFace model: {e}")
                self.llm = "mock"
        
        else:
            self.llm = "mock"
            print("Using mock LLM for demonstration")
    
    def retrieve(self, query: str) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query"""
        if not self.retriever:
            return []
        
        try:
            results = self.retriever.retrieve(query, self.top_k)
            return results
        except Exception as e:
            print(f"Error in retrieval: {e}")
            return []
    
    def generateAnswer(self, query: str, retrieved_docs: List[RetrievalResult]) -> str:
        """Generate an answer using the LLM based on retrieved documents"""
        if not retrieved_docs:
            return "I couldn't find any relevant documents to answer your question. Please try rephrasing your query."
        
        # Prepare context from retrieved documents
        context = self._prepareContext(retrieved_docs)
        
        # Generate answer based on LLM provider
        if self.llm == "openai":
            return self._generateOpenAIAnswer(query, context)
        elif self.llm == "huggingface":
            return self._generateHuggingFaceAnswer(query, context)
        else:
            return self._generateMockAnswer(query, context)
    
    def _prepareContext(self, retrieved_docs: List[RetrievalResult]) -> str:
        """Prepare context string from retrieved documents"""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs):
            # Truncate content if too long
            content = doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
            context_parts.append(f"Document {doc.document_id}:\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _generateOpenAIAnswer(self, query: str, context: str) -> str:
        """Generate answer using OpenAI API"""
        try:
            prompt = f"""Based on the following context, answer the question. 
            If the answer cannot be found in the context, say so.
            Always cite the document IDs you used in your answer.
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:"""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided documents. Always cite your sources."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return self._generateMockAnswer(query, context)
    
    def _generateHuggingFaceAnswer(self, query: str, context: str) -> str:
        """Generate answer using HuggingFace model"""
        try:
            prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the answer part
            answer_start = response.find("Answer:")
            if answer_start != -1:
                answer = response[answer_start + 7:].strip()
            else:
                answer = response
            
            return answer
        
        except Exception as e:
            print(f"Error generating HuggingFace answer: {e}")
            return self._generateMockAnswer(query, context)
    
    def _generateMockAnswer(self, query: str, context: str) -> str:
        """Generate a mock answer for demonstration purposes"""
        # Simple template-based answer generation
        doc_ids = []
        for line in context.split('\n'):
            if line.startswith('Document '):
                doc_id = line.split(':')[0].replace('Document ', '')
                doc_ids.append(doc_id)
        
        if doc_ids:
            citations = ", ".join([f"[{doc_id}]" for doc_id in doc_ids[:3]])
            answer = f"Based on the retrieved documents, I can provide information related to your query about '{query}'. "
            answer += f"The most relevant documents are: {citations}. "
            answer += "Please note that this is a mock response. For actual answers, ensure you have a proper LLM configured."
        else:
            answer = f"I found some relevant information for your query about '{query}', but I'm currently using a mock LLM. "
            answer += "To get real answers, please configure an OpenAI API key or HuggingFace model."
        
        return answer
    
    def processQuery(self, query: str) -> Dict[str, Any]:
        """Process a complete query through the RAG pipeline"""
        start_time = time.time()
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query)
        retrieval_time = time.time() - start_time
        
        # Generate answer
        generation_start = time.time()
        answer = self.generateAnswer(query, retrieved_docs)
        generation_time = time.time() - generation_start
        
        # Evaluate answer quality
        grounding_metrics = self.evaluator.evaluate_source_grounding(answer, retrieved_docs)
        factual_metrics = self.evaluator.evaluate_factual_correctness(answer, retrieved_docs)
        
        # Prepare results
        results = {
            "query": query,
            "answer": answer,
            "retrieved_documents": [
                {
                    "id": doc.document_id,
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "score": doc.score,
                    "metadata": doc.metadata
                }
                for doc in retrieved_docs
            ],
            "metrics": {
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": time.time() - start_time,
                "grounding_score": grounding_metrics["grounding_score"],
                "coverage": grounding_metrics["coverage"],
                "has_citations": factual_metrics["has_citations"]
            }
        }
        
        return results

class GradioRAGInterface:
    """Gradio interface for the RAG system"""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system
        self.chat_history = []
        
        # Create Gradio interface
        self.interface = self._createInterface()
    
    def _createInterface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="RAG Chat Interface", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🤖 RAG Chat Interface")
            gr.Markdown("Ask questions and get answers based on the Cranfield dataset!")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Chat interface
                    chatbot = gr.Chatbot(
                        label="Chat History",
                        height=500,
                        show_label=True
                    )
                    
                    with gr.Row():
                        query_input = gr.Textbox(
                            label="Ask a question",
                            placeholder="Enter your question here...",
                            lines=2
                        )
                        submit_btn = gr.Button("Submit", variant="primary")
                    
                    clear_btn = gr.Button("Clear Chat")
                
                with gr.Column(scale=1):
                    # System info and metrics
                    gr.Markdown("## 📊 System Information")
                    
                    retriever_info = gr.Textbox(
                        label="Retriever Type",
                        value=f"Type: {type(self.rag_system.retriever).__name__}",
                        interactive=False
                    )
                    
                    llm_info = gr.Textbox(
                        label="LLM Provider",
                        value=f"Provider: {self.rag_system.llm}",
                        interactive=False
                    )
                    
                    doc_count = gr.Textbox(
                        label="Documents Loaded",
                        value=f"Count: {len(self.rag_system.documents)}",
                        interactive=False
                    )
                    
                    # Metrics display
                    gr.Markdown("## 📈 Query Metrics")
                    metrics_display = gr.JSON(label="Latest Query Metrics")
            
            # Event handlers
            submit_btn.click(
                fn=self._handleQuery,
                inputs=[query_input, chatbot],
                outputs=[chatbot, metrics_display]
            )
            
            query_input.submit(
                fn=self._handleQuery,
                inputs=[query_input, chatbot],
                outputs=[chatbot, metrics_display]
            )
            
            clear_btn.click(
                fn=self._clearChat,
                inputs=[],
                outputs=[chatbot, metrics_display]
            )
        
        return interface
    
    def _handleQuery(self, query: str, chat_history: List[List[str]]) -> Tuple[List[List[str]], Dict]:
        """Handle a user query"""
        if not query.strip():
            return chat_history, {}
        
        # Process query through RAG system
        results = self.rag_system.processQuery(query)
        
        # Add to chat history
        chat_history.append([query, results["answer"]])
        
        # Return updated chat and metrics
        return chat_history, results["metrics"]
    
    def _clearChat(self) -> Tuple[List[List[str]], Dict]:
        """Clear the chat history"""
        return [], {}
    
    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        return self.interface.launch(**kwargs)

def createRAGSystem(config: Dict[str, Any] = None) -> RAGSystem:
    """Create a RAG system with the given configuration"""
    if config is None:
        config = {}
    
    # Default configuration
    default_config = {
        "dataset_path": "cranfield/",
        "retriever_type": "auto",
        "llm_provider": "openai",
        "top_k": 5,
        "chunk_size": 500
    }
    
    # Update with provided config
    default_config.update(config)
    
    return RAGSystem(**default_config)

def main():
    """Main function to run the RAG server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Server for Information Retrieval System")
    parser.add_argument("--dataset", default="cranfield/", help="Path to dataset folder")
    parser.add_argument("--retriever", default="auto", choices=["auto", "lsa", "faiss", "hybrid"], 
                       help="Type of retriever to use")
    parser.add_argument("--llm", default="openai", choices=["openai", "huggingface", "mock"], 
                       help="LLM provider to use")
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--chunk_size", type=int, default=500, help="Document chunk size")
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio interface")
    parser.add_argument("--share", action="store_true", help="Create public link for interface")
    
    args = parser.parse_args()
    
    # Create RAG system
    config = {
        "dataset_path": args.dataset,
        "retriever_type": args.retriever,
        "llm_provider": args.llm,
        "top_k": args.top_k,
        "chunk_size": args.chunk_size
    }
    
    print("Initializing RAG system...")
    rag_system = createRAGSystem(config)
    
    # Create and launch interface
    interface = GradioRAGInterface(rag_system)
    
    print(f"Launching RAG interface on port {args.port}")
    interface.launch(
        server_port=args.port,
        share=args.share,
        show_error=True
    )

if __name__ == "__main__":
    main()
