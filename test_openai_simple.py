#!/usr/bin/env python3
"""
Simple OpenAI Test
Tests if your new API key is working
"""

import os
import sys

def test_openai():
    """Test OpenAI connection"""
    print("🧪 Testing OpenAI Connection")
    print("=" * 40)
    
    try:
        # Import OpenAI
        import openai
        print("✓ OpenAI package imported successfully")
        
        # Get API key from config
        from config import get_openai_api_key
        api_key = get_openai_api_key()
        
        if api_key and api_key.startswith("sk-"):
            print(f"✓ API key found: {api_key[:10]}...")
            
            # Set API key
            openai.api_key = api_key
            
            # Test simple request
            print("\n🔍 Testing API connection...")
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Say 'Hello, RAG system is working!'"}
                ],
                max_tokens=20
            )
            
            answer = response.choices[0].message.content
            print(f"✓ OpenAI API working! Response: {answer}")
            print("\n🎉 Your API key is working perfectly!")
            return True
            
        else:
            print("✗ Invalid API key format")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Main test function"""
    success = test_openai()
    
    if success:
        print("\n✅ Next steps:")
        print("1. Restart RAG server: python run_rag_openai.py --llm openai")
        print("2. Ask questions and get real answers!")
    else:
        print("\n❌ Check your API key and try again")
    
    return success

if __name__ == "__main__":
    main()
