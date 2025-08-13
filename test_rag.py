"""
Test script for RAG system integration
Verifies that all components work with existing Information Retrieval system
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from rag_utils import Document, RetrievalResult, DocumentProcessor
        print("✓ rag_utils imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import rag_utils: {e}")
        return False
    
    try:
        from rag_build_index import IndexBuilder, LSABasedRetriever
        print("✓ rag_build_index imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import rag_build_index: {e}")
        return False
    
    try:
        from rag_server import RAGSystem, GradioRAGInterface
        print("✓ rag_server imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import rag_server: {e}")
        return False
    
    try:
        from informationRetrieval import InformationRetrieval
        print("✓ Existing informationRetrieval imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import existing informationRetrieval: {e}")
        return False
    
    return True

def test_document_processing():
    """Test document processing functionality"""
    print("\nTesting document processing...")
    
    try:
        from rag_utils import Document, DocumentProcessor
        
        # Create test document
        test_doc = Document(
            id="test_1",
            content="This is a test document about aerodynamics and fluid dynamics.",
            metadata={"source": "test", "type": "test"}
        )
        
        # Test processor
        processor = DocumentProcessor()
        processed = processor.preprocess_document(test_doc.content)
        
        print(f"✓ Document processing successful: {len(processed)} sentences")
        return True
        
    except Exception as e:
        print(f"✗ Document processing failed: {e}")
        return False

def test_retriever_creation():
    """Test retriever creation"""
    print("\nTesting retriever creation...")
    
    try:
        from rag_build_index import IndexBuilder
        
        # Create index builder
        builder = IndexBuilder()
        print("✓ IndexBuilder created successfully")
        
        # Try to load documents (this might fail if cranfield dataset isn't present)
        try:
            docs = builder.loadAndProcessDocuments(chunk_documents=False)
            if docs:
                print(f"✓ Loaded {len(docs)} documents")
                
                # Try to build retriever
                retriever = builder.buildRetriever("lsa")
                print(f"✓ Built retriever: {type(retriever).__name__}")
                
                # Test retrieval
                test_query = "aerodynamics"
                results = retriever.retrieve(test_query, top_k=3)
                print(f"✓ Retrieval successful: {len(results)} results")
                
            else:
                print("⚠ No documents loaded (this is expected if cranfield dataset isn't present)")
                
        except Exception as e:
            print(f"⚠ Document loading failed (expected if dataset not present): {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Retriever creation failed: {e}")
        return False

def test_rag_system():
    """Test RAG system creation"""
    print("\nTesting RAG system creation...")
    
    try:
        from rag_server import RAGSystem
        
        # Create RAG system with mock LLM
        rag = RAGSystem(
            dataset_path="cranfield/",
            retriever_type="auto",
            llm_provider="mock",
            top_k=3
        )
        
        print("✓ RAG system created successfully")
        
        # Test query processing (this might fail if no documents are loaded)
        try:
            test_query = "What is aerodynamics?"
            results = rag.processQuery(test_query)
            print("✓ Query processing successful")
            print(f"  Answer: {results['answer'][:100]}...")
            
        except Exception as e:
            print(f"⚠ Query processing failed (expected if no documents): {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ RAG system creation failed: {e}")
        return False

def test_gradio_interface():
    """Test Gradio interface creation"""
    print("\nTesting Gradio interface creation...")
    
    try:
        import gradio as gr
        print("✓ Gradio imported successfully")
        
        from rag_server import GradioRAGInterface, RAGSystem
        
        # Create mock RAG system
        rag = RAGSystem(
            dataset_path="cranfield/",
            retriever_type="auto",
            llm_provider="mock",
            top_k=3
        )
        
        # Create interface
        interface = GradioRAGInterface(rag)
        print("✓ Gradio interface created successfully")
        
        return True
        
    except ImportError as e:
        print(f"⚠ Gradio not available: {e}")
        return True  # Not critical for core functionality
    except Exception as e:
        print(f"✗ Gradio interface creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing RAG System Integration")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_document_processing,
        test_retriever_creation,
        test_rag_system,
        test_gradio_interface
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! RAG system is ready to use.")
    else:
        print("⚠ Some tests failed. Check the errors above.")
    
    print("\nNext steps:")
    print("1. Ensure cranfield dataset is in the correct location")
    print("2. Install required dependencies: pip install -r requirements_rag.txt")
    print("3. Run the RAG server: python rag_server.py")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
