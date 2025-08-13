"""
Minimal test script for RAG system
Tests core functionality without requiring all dependencies
"""

def test_basic_imports():
    """Test basic imports that should work"""
    print("Testing basic imports...")
    
    try:
        # Test if we can import existing modules
        from informationRetrieval import InformationRetrieval
        print("✓ Existing InformationRetrieval imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import existing InformationRetrieval: {e}")
        return False
    
    try:
        # Test if we can import basic Python modules
        import json
        import os
        print("✓ Basic Python modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import basic modules: {e}")
        return False
    
    return True

def test_rag_components():
    """Test RAG components with minimal dependencies"""
    print("\nTesting RAG components...")
    
    try:
        # Test if we can import RAG utilities
        from rag_utils import Document, RetrievalResult
        print("✓ RAG utility classes imported successfully")
        
        # Test creating a document
        doc = Document(
            id="test_1",
            content="This is a test document about aerodynamics.",
            metadata={"source": "test"}
        )
        print(f"✓ Document created: {doc.id}")
        
        # Test creating retrieval result
        result = RetrievalResult(
            document_id="test_1",
            content="Test content",
            score=0.8
        )
        print(f"✓ Retrieval result created: {result.document_id}")
        
        return True
        
    except ImportError as e:
        print(f"⚠ RAG components not available: {e}")
        print("This is expected if dependencies aren't installed yet.")
        return False

def test_existing_system():
    """Test that existing system still works"""
    print("\nTesting existing system...")
    
    try:
        from informationRetrieval import InformationRetrieval
        
        # Create instance
        ir = InformationRetrieval()
        print("✓ InformationRetrieval instance created")
        
        # Test basic functionality
        if hasattr(ir, 'buildIndex'):
            print("✓ buildIndex method available")
        if hasattr(ir, 'rank'):
            print("✓ rank method available")
        
        return True
        
    except Exception as e:
        print(f"✗ Existing system test failed: {e}")
        return False

def main():
    """Run minimal tests"""
    print("🧪 Minimal RAG System Test")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_rag_components,
        test_existing_system
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Core system is working.")
    else:
        print("⚠ Some tests failed. This is expected if dependencies aren't installed.")
    
    print("\nNext steps:")
    print("1. Install dependencies: pip3 install -r requirements_rag.txt")
    print("2. Run full test: python3 test_rag.py")
    print("3. Launch RAG server: python3 rag_server.py")
    
    return passed == total

if __name__ == "__main__":
    main()
