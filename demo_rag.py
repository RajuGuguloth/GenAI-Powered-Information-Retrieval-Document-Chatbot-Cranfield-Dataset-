"""
Demo script for RAG system
Shows how to use the RAG system programmatically
"""

import os
import sys
from pathlib import Path

def demo_basic_rag():
    """Demonstrate basic RAG functionality"""
    print("🚀 Basic RAG Demo")
    print("=" * 50)
    
    try:
        from rag_server import RAGSystem
        
        # Create RAG system with mock LLM for demonstration
        print("Creating RAG system...")
        rag = RAGSystem(
            dataset_path="cranfield/",
            retriever_type="auto",
            llm_provider="mock",  # Use mock LLM for demo
            top_k=3,
            chunk_size=500
        )
        
        if not rag.documents:
            print("⚠ No documents loaded. This is expected if cranfield dataset isn't present.")
            print("The demo will show the system structure but won't process queries.")
            return
        
        print(f"✓ RAG system created with {len(rag.documents)} documents")
        print(f"✓ Retriever type: {type(rag.retriever).__name__}")
        print(f"✓ LLM provider: {rag.llm}")
        
        # Demo queries
        demo_queries = [
            "What is aerodynamics?",
            "Explain fluid dynamics",
            "What are the applications of computational fluid dynamics?"
        ]
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n📝 Query {i}: {query}")
            print("-" * 40)
            
            try:
                results = rag.processQuery(query)
                
                print(f"Answer: {results['answer']}")
                print(f"Retrieved {len(results['retrieved_documents'])} documents")
                
                # Show metrics
                metrics = results['metrics']
                print(f"Retrieval time: {metrics['retrieval_time']:.3f}s")
                print(f"Generation time: {metrics['generation_time']:.3f}s")
                print(f"Grounding score: {metrics['grounding_score']:.3f}")
                
            except Exception as e:
                print(f"⚠ Query processing failed: {e}")
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        print("Make sure you have installed the required dependencies:")
        print("pip install -r requirements_rag.txt")

def demo_custom_documents():
    """Demonstrate using custom documents"""
    print("\n\n📚 Custom Documents Demo")
    print("=" * 50)
    
    try:
        from rag_utils import Document, create_document_from_text
        from rag_build_index import LSABasedRetriever
        
        # Create custom documents
        custom_docs = [
            Document(
                id="custom_1",
                content="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                metadata={"topic": "AI/ML", "source": "custom"}
            ),
            Document(
                id="custom_2", 
                content="Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
                metadata={"topic": "AI/ML", "source": "custom"}
            ),
            Document(
                id="custom_3",
                content="Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and human language.",
                metadata={"topic": "AI/ML", "source": "custom"}
            )
        ]
        
        print(f"✓ Created {len(custom_docs)} custom documents")
        
        # Create retriever
        retriever = LSABasedRetriever(custom_docs)
        print("✓ Built LSA-based retriever")
        
        # Test retrieval
        test_query = "What is machine learning?"
        results = retriever.retrieve(test_query, top_k=2)
        
        print(f"\nQuery: {test_query}")
        print(f"Retrieved {len(results)} documents:")
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. Doc {result.document_id} (Score: {result.score:.3f})")
            print(f"     Content: {result.content[:100]}...")
        
    except Exception as e:
        print(f"✗ Custom documents demo failed: {e}")

def demo_evaluation():
    """Demonstrate RAG evaluation capabilities"""
    print("\n\n📊 Evaluation Demo")
    print("=" * 50)
    
    try:
        from rag_utils import RAGEvaluator, RetrievalResult
        
        evaluator = RAGEvaluator()
        
        # Mock retrieval results
        mock_results = [
            RetrievalResult(
                document_id="doc1",
                content="Machine learning algorithms can process large amounts of data to identify patterns.",
                score=0.9,
                metadata={"source": "mock"}
            ),
            RetrievalResult(
                document_id="doc2", 
                content="Deep learning is a subset of machine learning that uses neural networks.",
                score=0.8,
                metadata={"source": "mock"}
            )
        ]
        
        # Mock answer
        mock_answer = "Machine learning is a field of AI that uses algorithms to process data and identify patterns. Deep learning is a subset that uses neural networks."
        
        # Evaluate
        grounding_metrics = evaluator.evaluate_source_grounding(mock_answer, mock_results)
        factual_metrics = evaluator.evaluate_factual_correctness(mock_answer, mock_results)
        
        print("Mock Answer:", mock_answer)
        print("\nGrounding Metrics:")
        for key, value in grounding_metrics.items():
            print(f"  {key}: {value}")
        
        print("\nFactual Metrics:")
        for key, value in factual_metrics.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"✗ Evaluation demo failed: {e}")

def demo_advanced_features():
    """Demonstrate advanced RAG features"""
    print("\n\n🔧 Advanced Features Demo")
    print("=" * 50)
    
    try:
        from rag_build_index import IndexBuilder, HybridRetriever
        
        # Show index builder capabilities
        print("Index Builder Features:")
        builder = IndexBuilder()
        print("✓ Can load documents from multiple sources")
        print("✓ Automatic document chunking")
        print("✓ Multiple retriever types")
        
        # Show retriever options
        print("\nRetriever Options:")
        print("  - LSA-based (reuses your existing system)")
        print("  - FAISS-based (if available)")
        print("  - Hybrid (combines both)")
        
        # Show LLM options
        print("\nLLM Options:")
        print("  - OpenAI (requires API key)")
        print("  - HuggingFace (local models)")
        print("  - Mock (for testing)")
        
    except Exception as e:
        print(f"✗ Advanced features demo failed: {e}")

def main():
    """Run all demos"""
    print("🎭 RAG System Demo")
    print("=" * 60)
    
    demos = [
        demo_basic_rag,
        demo_custom_documents,
        demo_evaluation,
        demo_advanced_features
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"Demo failed: {e}")
        
        print("\n" + "=" * 60)
    
    print("\n🎯 Demo Complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements_rag.txt")
    print("2. Test the system: python test_rag.py")
    print("3. Launch the interface: python rag_server.py")
    print("4. Check the README: README_RAG.md")

if __name__ == "__main__":
    main()
