#!/usr/bin/env python3
"""
Test PayPal integration with the correct flow
"""
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def get_paypal_token():
    """Step 1: Get PayPal access token"""
    client_id = os.getenv("PAYPAL_CLIENT_ID")
    client_secret = os.getenv("PAYPAL_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("❌ PayPal credentials not found in .env")
        return None
    
    print(f"✅ Client ID: {client_id[:10]}...")
    
    url = "https://api-m.sandbox.paypal.com/v1/oauth2/token"
    data = 'grant_type=client_credentials'
    
    try:
        response = requests.post(
            url, 
            data=data,
            auth=(client_id, client_secret)
        )
        response.raise_for_status()
        token_data = response.json()
        access_token = token_data.get('access_token')
        
        print(f"✅ Access token obtained: {access_token[:20]}...")
        return access_token
        
    except Exception as e:
        print(f"❌ Failed to get access token: {e}")
        return None

def create_invoice(access_token, recipient_email="achihsingh@gmail.com"):
    """Step 2: Create an Invoice"""
    print(f"\n📧 Creating PayPal invoice for {recipient_email}...")
    
    url = "https://api-m.sandbox.paypal.com/v2/invoicing/invoices"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {access_token}',
    }
    
    invoice_data = {
        "detail": {
            "currency_code": "USD",
            "note": "Thanks for using our voice assistant service!"
        },
        "invoicer": {
            "name": {
                "given_name": "Achintya",
                "surname": "Singh"
            }
        },
        "primary_recipients": [
            {
                "billing_info": {
                    "email_address": recipient_email
                }
            }
        ],
        "items": [
            {
                "name": "Payment Reminder Test",
                "quantity": "1",
                "unit_amount": {
                    "currency_code": "USD",
                    "value": "25.00"
                }
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=invoice_data)
        print(f"Create response status: {response.status_code}")
        
        if response.status_code == 201:
            response_data = response.json()
            # Extract invoice ID from href URL
            href = response_data.get('href', '')
            if href:
                invoice_id = href.split('/')[-1]
                print(f"✅ Invoice created: {invoice_id}")
                return invoice_id
            else:
                print("❌ Could not extract invoice ID from response")
                return None
        else:
            print(f"❌ Failed to create invoice: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to create invoice: {e}")
        return None

def send_invoice(access_token, invoice_id):
    """Step 3: Send the Invoice"""
    print(f"\n📤 Sending invoice: {invoice_id}")
    
    url = f"https://api-m.sandbox.paypal.com/v2/invoicing/invoices/{invoice_id}/send"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {access_token}',
    }
    
    try:
        response = requests.post(url, headers=headers)
        print(f"Send response status: {response.status_code}")
        
        if response.status_code in [200, 202]:
            print("✅ Invoice sent successfully!")
            return True
        else:
            print(f"❌ Failed to send invoice: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to send invoice: {e}")
        return False

def send_reminder(access_token, invoice_id):
    """Step 4: Send a Reminder"""
    print(f"\n📨 Sending payment reminder for invoice: {invoice_id}")
    
    url = f"https://api-m.sandbox.paypal.com/v2/invoicing/invoices/{invoice_id}/remind"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {access_token}',
    }
    
    reminder_data = {
        "subject": "Reminder: Your invoice is still unpaid",
        "note": "Hi! This is a test payment reminder from your voice assistant. Please complete your payment when convenient. Thanks!"
    }
    
    try:
        response = requests.post(url, headers=headers, json=reminder_data)
        print(f"Reminder response status: {response.status_code}")
        
        if response.status_code == 204:
            print("✅ Payment reminder sent successfully!")
            print("📧 Reminder email sent via PayPal")
            return True
        else:
            print(f"❌ Failed to send reminder: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to send reminder: {e}")
        return False

def main():
    print("🧪 Testing PayPal Integration (Correct Flow)...\n")
    
    # Step 1: Get access token
    print("=== Step 1: Get Access Token ===")
    token = get_paypal_token()
    if not token:
        return
    
    # Step 2: Create invoice
    print("\n=== Step 2: Create Invoice ===")
    invoice_id = create_invoice(token)
    if not invoice_id:
        return
    
    # Step 3: Send invoice
    print("\n=== Step 3: Send Invoice ===")
    sent = send_invoice(token, invoice_id)
    if not sent:
        return
    
    # Step 4: Send reminder
    print("\n=== Step 4: Send Reminder ===")
    success = send_reminder(token, invoice_id)
    
    if success:
        print("\n🎉 SUCCESS! Complete PayPal flow working!")
        print("📧 Invoice and reminder sent via PayPal")
        print("🔍 Check PayPal sandbox notifications")
        print(f"📋 Invoice ID: {invoice_id}")
        print("\nReady to integrate into voice assistant!")
    else:
        print("\n❌ Something went wrong. Check the errors above.")

if __name__ == "__main__":
    main()