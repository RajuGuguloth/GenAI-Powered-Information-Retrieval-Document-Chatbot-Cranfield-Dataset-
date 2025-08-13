#!/usr/bin/env python3
"""
RAG Server Launcher with OpenAI Integration
Automatically launches the RAG server with your OpenAI API key
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch RAG server with OpenAI"""
    print("🚀 Launching RAG Server with OpenAI Integration")
    print("=" * 60)
    
    # Check if config exists
    config_file = Path("config.py")
    if not config_file.exists():
        print("✗ Config file not found. Please ensure config.py exists.")
        return False
    
    # Import config to set API key
    try:
        from config import get_openai_api_key
        api_key = get_openai_api_key()
        
        if api_key and api_key.startswith("sk-"):
            print("✓ OpenAI API key loaded successfully")
            print(f"  Key: {api_key[:10]}...")
            
            # Set environment variable
            os.environ["OPENAI_API_KEY"] = api_key
            print("✓ Environment variable set")
            
        else:
            print("✗ Invalid API key format")
            return False
            
    except ImportError as e:
        print(f"✗ Failed to import config: {e}")
        return False
    
    # Check if rag_server.py exists
    server_file = Path("rag_server.py")
    if not server_file.exists():
        print("✗ rag_server.py not found")
        return False
    
    print("\n🎯 Starting RAG Server...")
    print("  - LLM Provider: OpenAI")
    print("  - Retriever: Auto-detect (LSA-based)")
    print("  - Interface: Gradio web UI")
    print("  - Port: 7860")
    
    print("\n📱 The web interface will open at: http://localhost:7860")
    print("🔄 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Launch the RAG server
        cmd = [sys.executable, "rag_server.py", "--llm", "openai"]
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n✗ Error launching server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
