#!/usr/bin/env python3
"""
Simple test to verify basic webhook functionality
"""
from fastapi import FastAPI, Request
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Server is running!"}

@app.post("/test")
async def simple_test():
    """Ultra simple test endpoint"""
    print("🎉 TEST ENDPOINT HIT!")
    return {"status": "success", "message": "Webhook is working!"}

@app.post("/incoming_call")
async def handle_call(request: Request):
    """Simple Twilio webhook"""
    print("📞 TWILIO CALL RECEIVED!")
    
    # Log request details
    print(f"Headers: {dict(request.headers)}")
    print(f"Client IP: {request.client.host if request.client else 'Unknown'}")
    
    try:
        form_data = await request.form()
        print(f"Form data: {dict(form_data)}")
    except Exception as e:
        print(f"Error reading form: {e}")
    
    # Create simple TwiML response
    response = VoiceResponse()
    response.say("Hello! This is a test. Your webhook is working correctly!")
    
    xml_response = str(response)
    print(f"Sending TwiML: {xml_response}")
    
    # Add headers to help with any CORS issues
    return Response(
        content=xml_response, 
        media_type="application/xml",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

if __name__ == "__main__":
    print("🚀 Starting simple test server...")
    print("Test endpoints:")
    print("  GET  /     - Basic health check")
    print("  POST /test - Simple test endpoint") 
    print("  POST /incoming_call - Twilio webhook")
    print("\nServer will run on http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)