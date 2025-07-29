#!/usr/bin/env python3
"""
Test script to verify Twilio integration setup
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_environment():
    """Test if all required environment variables are set"""
    required_vars = [
        'SARVAM_API_KEY',
        'TWILIO_ACCOUNT_SID', 
        'TWILIO_AUTH_TOKEN',
        'SPLITWISE_API_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        return False
    else:
        print("✅ All environment variables are set")
        return True

def test_server_running():
    """Test if the FastAPI server is running"""
    try:
        response = requests.get("http://localhost:8000/api/health", timeout=5)
        if response.status_code == 200:
            print("✅ FastAPI server is running")
            return True
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Server is not running: {e}")
        return False

def test_twilio_credentials():
    """Test Twilio credentials"""
    try:
        from twilio.rest import Client
        
        account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        
        client = Client(account_sid, auth_token)
        account = client.api.accounts(account_sid).fetch()
        
        print(f"✅ Twilio credentials valid - Account: {account.friendly_name}")
        return True
    except Exception as e:
        print(f"❌ Twilio credentials invalid: {e}")
        return False

def test_sarvam_api():
    """Test SarvamAI API"""
    try:
        from sarvamai import SarvamAI
        
        api_key = os.getenv('SARVAM_API_KEY')
        client = SarvamAI(api_subscription_key=api_key)
        
        print("✅ SarvamAI client initialized successfully")
        return True
    except Exception as e:
        print(f"❌ SarvamAI initialization failed: {e}")
        return False

def main():
    print("🔍 Testing Twilio Integration Setup...\n")
    
    tests = [
        ("Environment Variables", test_environment),
        ("FastAPI Server", test_server_running),
        ("Twilio Credentials", test_twilio_credentials),
        ("SarvamAI API", test_sarvam_api)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        result = test_func()
        results.append(result)
        print()
    
    if all(results):
        print("🎉 All tests passed! Your setup is ready for Twilio integration.")
        print("\nNext steps:")
        print("1. Start ngrok: ngrok http 8000")
        print("2. Update the WebSocket URL in main.py with your ngrok URL")
        print("3. Configure your Twilio phone number webhook")
        print("4. Test by calling your Twilio number")
    else:
        print("❌ Some tests failed. Please fix the issues above before proceeding.")

if __name__ == "__main__":
    main()