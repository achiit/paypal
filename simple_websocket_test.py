#!/usr/bin/env python3
"""
Simple WebSocket test server for Twilio
"""
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse, Connect
import uvicorn
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/incoming_call")
async def handle_call(request: Request):
    """Handle incoming Twilio call"""
    logger.info("📞 Incoming call received")
    
    response = VoiceResponse()
    response.say("Hello! Testing WebSocket connection.", voice='alice')
    
    connect = Connect()
    connect.stream(url="wss://5fd4c229db5f.ngrok-free.app/ws")
    response.append(connect)
    
    logger.info(f"Sending TwiML: {str(response)}")
    return Response(content=str(response), media_type="application/xml")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Simple WebSocket handler for Twilio"""
    try:
        # Log connection details for debugging
        logger.info(f"🔌 WebSocket connection attempt from: {websocket.client}")
        logger.info(f"🔌 Headers: {dict(websocket.headers)}")
        
        await websocket.accept()
        logger.info("🔌 WebSocket connection accepted from Twilio")
        
        # Send a keep-alive message to confirm connection
        await websocket.send_json({"type": "connection_established"})
        
        while True:
            try:
                # Receive message from Twilio with timeout
                message = await websocket.receive_json()
                event = message.get("event")
                
                logger.info(f"📨 Received event: {event}")
                logger.info(f"📨 Full message: {json.dumps(message, indent=2)}")
                
                if event == "start":
                    stream_sid = message["start"]["streamSid"]
                    logger.info(f"� Steream started: {stream_sid}")
                    
                    # Send acknowledgment back to Twilio
                    await websocket.send_json({
                        "event": "start_ack",
                        "streamSid": stream_sid
                    })
                    
                elif event == "media":
                    # Just log that we received media, don't process it yet
                    logger.info("🎤 Received audio data")
                    
                elif event == "stop":
                    logger.info("🛑 Stream stopped")
                    break
                    
            except Exception as e:
                logger.error(f"❌ Error processing message: {e}")
                break
                
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
    finally:
        logger.info("🔌 WebSocket connection closed")

@app.get("/")
async def root():
    return {"message": "Simple WebSocket test server running"}

if __name__ == "__main__":
    print("🚀 Starting simple WebSocket test server...")
    print("Endpoints:")
    print("  POST /incoming_call - Twilio webhook")
    print("  WS   /ws - WebSocket endpoint")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)