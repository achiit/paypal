#!/usr/bin/env python3
"""
Test WebSocket connectivity for Twilio
"""
import asyncio
import websockets
import json
import base64

async def test_websocket():
    """Test WebSocket connection"""
    uri = "wss://5fd4c229db5f.ngrok-free.app/ws"
    
    try:
        print(f"🔍 Testing WebSocket connection to: {uri}")
        
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected successfully!")
            
            # Send a test message similar to what Twilio sends
            test_message = {
                "event": "start",
                "start": {
                    "streamSid": "test-stream-123",
                    "accountSid": "test-account",
                    "callSid": "test-call"
                }
            }
            
            await websocket.send(json.dumps(test_message))
            print("✅ Sent test start message")
            
            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"✅ Received response: {response}")
            except asyncio.TimeoutError:
                print("⚠️  No response received (this might be normal)")
            
            print("✅ WebSocket test completed successfully")
            
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(test_websocket())