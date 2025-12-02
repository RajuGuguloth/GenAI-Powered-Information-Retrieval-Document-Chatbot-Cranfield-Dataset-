# RAG (Retrieval-Augmented Generation) Integration

This directory contains the RAG integration for your existing Information Retrieval system. The RAG system enhances your LSA-based retrieval with LLM-powered answer generation.

##  Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_rag.txt
```

### 2. Run the RAG Server

```bash
python rag_server.py
```

This will launch a Gradio web interface at `http://localhost:7860`

##  Files Overview

- **`rag_utils.py`** - Core utilities for document processing, dataset loading, and evaluation
- **`rag_build_index.py`** - Index building and retrieval functionality (LSA + FAISS)
- **`rag_server.py`** - Main RAG server with Gradio chat interface
- **`test_rag.py`** - Test script to verify system functionality
- **`requirements_rag.txt`** - Python dependencies for RAG functionality

##  Configuration Options

### Command Line Arguments

```bash
python rag_server.py [OPTIONS]

Options:
  --dataset PATH        Path to dataset folder (default: cranfield/)
  --retriever TYPE      Retriever type: auto, lsa, faiss, hybrid (default: auto)
  --llm PROVIDER        LLM provider: openai, huggingface, mock (default: openai)
  --top_k INT           Number of documents to retrieve (default: 5)
  --chunk_size INT      Document chunk size (default: 500)
  --port INT            Port for Gradio interface (default: 7860)
  --share               Create public link for interface
```

### Environment Variables

For OpenAI integration:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

##  System Architecture

### 1. Document Processing
- **Reuses your existing preprocessing pipeline**: sentence segmentation, tokenization, lemmatization, stopword removal
- **Automatic document chunking** for better retrieval
- **Support for Cranfield dataset** and fallback to text files

### 2. Retrieval Layer
- **LSA-based retrieval** (reuses your existing `InformationRetrieval` class)
- **FAISS-based retrieval** (optional, using sentence transformers)
- **Hybrid retrieval** combining both approaches
- **Automatic detection** of existing retriever functions

### 3. Generation Layer
- **OpenAI GPT models** (GPT-3.5, GPT-4)
- **HuggingFace models** (local inference)
- **Mock LLM** for testing and demonstration
- **Source citation** in generated answers

### 4. Evaluation
- **Source grounding metrics** (keyword overlap, coverage)
- **Factual correctness** evaluation
- **Performance metrics** (retrieval time, generation time)

## 🔍 Usage Examples

### Basic Usage

```python
from rag_server import RAGSystem

# Create RAG system
rag = RAGSystem(
    dataset_path="cranfield/",
    retriever_type="auto",
    llm_provider="mock"  # Use "openai" for real LLM
)

# Process a query
results = rag.processQuery("What is aerodynamics?")
print(results["answer"])
```

### Custom Retriever

```python
from rag_build_index import LSABasedRetriever, Document

# Create custom retriever
documents = [Document(id="1", content="Your document content")]
retriever = LSABasedRetriever(documents)

# Use in RAG system
rag = RAGSystem(retriever=retriever)
```

### LangChain Integration

```python
from rag_build_index import LangChainRetrieverWrapper

# Wrap your retriever for LangChain
langchain_retriever = LangChainRetrieverWrapper(retriever, top_k=5)

# Use with LangChain chains
from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(
    llm=your_llm,
    retriever=langchain_retriever
)
```

##  Evaluation

### RAG-Specific Metrics

The system automatically evaluates:

- **Grounding Score**: How well the answer is grounded in retrieved documents
- **Coverage**: Percentage of retrieved document content used in the answer
- **Citation Presence**: Whether the answer cites source documents
- **Performance**: Retrieval and generation timing

### Integration with Existing Metrics

Your existing evaluation scripts remain unchanged. The RAG system adds new evaluation capabilities without disrupting the current pipeline.

##  Troubleshooting

### Common Issues

1. **"No documents loaded"**
   - Ensure Cranfield dataset is in the correct location
   - Check file paths in `cranfield/cran_docs.json`

2. **"OpenAI API key not found"**
   - Set `OPENAI_API_KEY` environment variable
   - Or use `--llm mock` for testing

3. **Import errors**
   - Install dependencies: `pip install -r requirements_rag.txt`
   - Check Python version compatibility

4. **Retrieval failures**
   - Verify existing `InformationRetrieval` class works
   - Check document preprocessing pipeline

### Testing

Run the test script to verify system functionality:

```bash
python test_rag.py
```

##  Integration with Existing System

### Minimal Disruption

- **No changes** to your existing evaluation scripts
- **Reuses** your preprocessing pipeline and LSA implementation
- **Adds** RAG functionality as a new layer
- **Maintains** all existing functionality

### File Structure

```
Main-Project/Code/
├── informationRetrieval.py    # Your existing IR system
├── main_VSM-1.py             # Your existing main script
├── evaluation.py              # Your existing evaluation
├── rag_utils.py              # NEW: RAG utilities
├── rag_build_index.py        # NEW: RAG retrieval
├── rag_server.py             # NEW: RAG server
├── test_rag.py               # NEW: RAG tests
└── requirements_rag.txt      # NEW: RAG dependencies
```

##  Advanced Features

### 1. Custom Document Sources

Add your own documents:

```python
from rag_utils import create_document_from_text

# Create document from text
doc = create_document_from_text("Your document content", "custom_id")

# Add to RAG system
rag.documents.append(doc)
```

### 2. Multiple Retrievers

Use different retrieval strategies:

```python
# LSA only
rag = RAGSystem(retriever_type="lsa")

# FAISS only (if available)
rag = RAGSystem(retriever_type="faiss")

# Hybrid approach
rag = RAGSystem(retriever_type="hybrid")
```

### 3. Custom LLM Prompts

Modify the prompt template in `rag_server.py`:

```python
def _generateOpenAIAnswer(self, query: str, context: str) -> str:
    prompt = f"""Your custom prompt template:
    Context: {context}
    Question: {query}
    Answer:"""
    # ... rest of the method
```

##  Performance Optimization

### 1. Document Chunking

Adjust chunk size for your use case:

```python
rag = RAGSystem(chunk_size=300)  # Smaller chunks for precise retrieval
rag = RAGSystem(chunk_size=800)  # Larger chunks for context
```

### 2. Retrieval Count

Balance between relevance and context:

```python
rag = RAGSystem(top_k=3)   # Fewer docs, more focused
rag = RAGSystem(top_k=10)  # More docs, broader context
```

### 3. Caching

The system automatically caches:
- Document embeddings (FAISS)
- Processed document content
- LLM responses (if supported)

##  Security Considerations

### API Keys

- Never commit API keys to version control
- Use environment variables for sensitive data
- Consider using `.env` files for local development

### Data Privacy

- Local HuggingFace models keep data on your machine
- OpenAI API sends data to external servers
- Mock LLM processes everything locally

## 📚 Further Reading

- [Gradio Documentation](https://gradio.app/docs/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

##  Support

For issues or questions:

1. Check the troubleshooting section above
2. Run `python test_rag.py` to identify problems
3. Verify your existing system works independently
4. Check dependency versions match requirements

##  Next Steps

1. **Test the system**: `python test_rag.py`
2. **Launch the interface**: `python rag_server.py`
3. **Configure LLM**: Set up OpenAI API key or HuggingFace model
4. **Customize**: Modify prompts, add custom documents
5. **Evaluate**: Use built-in metrics to assess performance

The RAG system is designed to enhance your existing Information Retrieval capabilities while maintaining full backward compatibility. Enjoy exploring the enhanced search and answer generation capabilities!
