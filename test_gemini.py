#!/usr/bin/env python3
"""
Test Gemini integration
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def test_gemini():
    """Test Gemini API"""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY not found in .env")
            return False
            
        print(f"✅ API Key found: {api_key[:10]}...")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        print("🔍 Testing Gemini with simple prompt...")
        
        response = model.generate_content(
            "Hello! Please respond with 'Gemini is working correctly!'",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=50,
                temperature=0.1,
            )
        )
        
        print(f"✅ Gemini response: {response.text}")
        return True
        
    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Gemini Integration...")
    success = test_gemini()
    
    if success:
        print("🎉 Gemini is ready to use!")
    else:
        print("❌ Fix the issues above before proceeding.")