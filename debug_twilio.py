#!/usr/bin/env python3
"""
Debug script for Twilio integration issues
"""
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def test_webhook_endpoint():
    """Test the incoming_call webhook endpoint"""
    print("🔍 Testing webhook endpoint...")
    
    try:
        # Test the webhook endpoint directly
        response = requests.post("http://localhost:8000/incoming_call", timeout=10)
        print(f"✅ Webhook responded with status: {response.status_code}")
        print(f"Response content: {response.text[:200]}...")
        
        # Check if it's valid TwiML
        if "VoiceResponse" in response.text or "<Response>" in response.text:
            print("✅ Response contains TwiML")
        else:
            print("❌ Response doesn't look like TwiML")
            
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Webhook test failed: {e}")
        return False

def test_ngrok_webhook():
    """Test the webhook through ngrok"""
    print("\n🔍 Testing webhook through ngrok...")
    
    # Extract ngrok URL from main.py
    try:
        with open("main.py", "r") as f:
            content = f.read()
            
        # Find the ngrok URL in the code
        import re
        pattern = r'wss://([^/]+)/ws'
        match = re.search(pattern, content)
        
        if match:
            ngrok_domain = match.group(1)
            webhook_url = f"https://{ngrok_domain}/incoming_call"
            
            print(f"Found ngrok URL: {webhook_url}")
            
            # Test the webhook through ngrok
            response = requests.post(webhook_url, timeout=10)
            print(f"✅ Ngrok webhook responded with status: {response.status_code}")
            print(f"Response content: {response.text[:200]}...")
            
            return response.status_code == 200
        else:
            print("❌ Could not find ngrok URL in main.py")
            return False
            
    except Exception as e:
        print(f"❌ Ngrok webhook test failed: {e}")
        return False

def test_twilio_phone_number():
    """Check Twilio phone number configuration"""
    print("\n🔍 Testing Twilio phone number configuration...")
    
    try:
        from twilio.rest import Client
        
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        
        client = Client(account_sid, auth_token)
        
        # Get all phone numbers
        phone_numbers = client.incoming_phone_numbers.list()
        
        if not phone_numbers:
            print("❌ No phone numbers found in your Twilio account")
            return False
            
        for number in phone_numbers:
            print(f"📞 Phone Number: {number.phone_number}")
            print(f"   Voice URL: {number.voice_url}")
            print(f"   Voice Method: {number.voice_method}")
            print(f"   Status: {number.status}")
            
            # Check if webhook is configured
            if number.voice_url:
                print("✅ Webhook URL is configured")
            else:
                print("❌ No webhook URL configured")
                
        return True
        
    except Exception as e:
        print(f"❌ Twilio phone number test failed: {e}")
        return False

def check_server_logs():
    """Instructions for checking server logs"""
    print("\n📋 To debug further, check your server logs:")
    print("1. Make sure your FastAPI server is running with:")
    print("   uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
    print("2. Watch the logs when you make a call")
    print("3. You should see 'Incoming call received' in the logs")
    print("4. If you don't see this message, the webhook isn't reaching your server")

def main():
    print("🚨 Twilio 'Busy' Signal Debugger\n")
    
    tests = [
        ("Local Webhook", test_webhook_endpoint),
        ("Ngrok Webhook", test_ngrok_webhook), 
        ("Twilio Phone Config", test_twilio_phone_number)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Testing: {test_name}")
        print('='*50)
        result = test_func()
        results.append(result)
    
    print(f"\n{'='*50}")
    print("SUMMARY")
    print('='*50)
    
    if all(results):
        print("✅ All tests passed!")
        print("\nIf you're still getting 'busy', try:")
        print("1. Restart your FastAPI server")
        print("2. Restart ngrok")
        print("3. Update Twilio webhook URL again")
        print("4. Wait 1-2 minutes for Twilio to update")
    else:
        print("❌ Some tests failed. Check the issues above.")
        
    check_server_logs()

if __name__ == "__main__":
    main()