#!/usr/bin/env python3
"""
Simple Twilio webhook with Gemini LLM integration
"""
from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse
import google.generativeai as genai
import os
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI()

# Initialize Gemini
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    print("✅ Gemini initialized successfully")
except Exception as e:
    print(f"❌ Gemini initialization failed: {e}")
    gemini_model = None

def get_gemini_response(user_input: str) -> str:
    """Get response from Gemini"""
    if not gemini_model:
        return "AI is currently unavailable."
    
    try:
        prompt = f"""
You are a helpful financial assistant. Keep responses short and conversational.

User said: "{user_input}"

Respond naturally and helpfully in 1-2 sentences.
"""
        
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=100,
                temperature=0.7,
            )
        )
        
        return response.text.strip()
        
    except Exception as e:
        print(f"Gemini error: {e}")
        return "I'm having trouble processing that right now."

@app.post("/incoming_call")
async def handle_call():
    """Handle incoming Twilio call with Gemini response"""
    print("📞 Call received!")
    
    # For now, let's use a simple greeting with Gemini
    user_input = "Hello, how are you?"
    gemini_response = get_gemini_response(user_input)
    
    print(f"🤖 Gemini says: {gemini_response}")
    
    response = VoiceResponse()
    response.say(f"Hello! {gemini_response}", voice='alice')
    
    xml_response = str(response)
    print(f"Sending: {xml_response}")
    
    return Response(content=xml_response, media_type="application/xml")

@app.get("/")
async def root():
    return {"message": "Twilio + Gemini server is running"}

@app.get("/test_gemini")
async def test_gemini_endpoint():
    """Test Gemini directly"""
    response = get_gemini_response("Hello, how are you?")
    return {"gemini_response": response}

if __name__ == "__main__":
    print("🚀 Starting Twilio + Gemini server...")
    print("Endpoints:")
    print("  POST /incoming_call - Twilio webhook")
    print("  GET  /test_gemini - Test Gemini directly")
    uvicorn.run(app, host="0.0.0.0", port=8000)