#!/usr/bin/env python3
"""
Super simple Twilio webhook - just says "Hello how are you"
"""
from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse
import uvicorn

app = FastAPI()

@app.post("/incoming_call")
async def handle_call():
    """Handle incoming Twilio call - just say hello"""
    print("📞 Call received!")
    
    response = VoiceResponse()
    response.say("Hello! How are you? This is working perfectly!", voice='alice')
    
    xml_response = str(response)
    print(f"Sending: {xml_response}")
    
    return Response(content=xml_response, media_type="application/xml")

@app.get("/")
async def root():
    return {"message": "Twilio webhook server is running"}

if __name__ == "__main__":
    print("🚀 Starting simple Twilio server...")
    print("Webhook URL: /incoming_call")
    uvicorn.run(app, host="0.0.0.0", port=8000)