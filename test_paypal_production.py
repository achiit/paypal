#!/usr/bin/env python3
"""
Test PayPal integration with PRODUCTION (real emails)
WARNING: This will send real emails!
"""
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def get_paypal_token_production():
    """Get PayPal access token for PRODUCTION"""
    client_id = "ASt_VkFP9FmEFrzTFpdLQuFrqxIPEL0Rb2A5M-gxTx2I1uNumrWsNjujv5v9gwlSxDdBSVPkgPXaodaB"
    client_secret = "EHj-zKvjv8cnHvmJANbsyjrdURQLHlaolbcAEUFV3rK0tRVY9SYl5VpDVCSjVxejG8Mi8vHu9F1G2VjD"
    
    if not client_id or not client_secret:
        print("❌ PayPal credentials not found in .env")
        return None
    
    print(f"⚠️  PRODUCTION MODE - Real emails will be sent!")
    print(f"✅ Client ID: {client_id[:10]}...")
    
    # PRODUCTION URL (not sandbox)
    url = "https://api-m.paypal.com/v1/oauth2/token"
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
        
        print(f"✅ PRODUCTION Access token obtained: {access_token[:20]}...")
        return access_token
        
    except Exception as e:
        print(f"❌ Failed to get production access token: {e}")
        print("💡 Your sandbox credentials won't work in production")
        print("💡 You need production PayPal app credentials")
        return None

def create_production_invoice(access_token, recipient_email="tenctacion23@gmail.com"):
    """Create PRODUCTION invoice (real email will be sent)"""
    print(f"\n📧 Creating PRODUCTION PayPal invoice for {recipient_email}...")
    print("⚠️  This will send a REAL email!")
    
    # PRODUCTION URL (not sandbox)
    url = "https://api-m.paypal.com/v2/invoicing/invoices"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {access_token}',
    }
    
    invoice_data = {
        "detail": {
            "currency_code": "USD",
            "note": "Test payment reminder from voice assistant (PRODUCTION TEST)"
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
                "name": "Voice Assistant Test",
                "quantity": "1",
                "unit_amount": {
                    "currency_code": "USD",
                    "value": "1.00"  # Small amount for testing
                }
            }
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=invoice_data)
        print(f"Create response status: {response.status_code}")
        
        if response.status_code == 201:
            response_data = response.json()
            href = response_data.get('href', '')
            if href:
                invoice_id = href.split('/')[-1]
                print(f"✅ PRODUCTION Invoice created: {invoice_id}")
                return invoice_id
        
        print(f"❌ Failed to create production invoice: {response.text}")
        return None
            
    except Exception as e:
        print(f"❌ Failed to create production invoice: {e}")
        return None

def send_production_invoice(access_token, invoice_id):
    """Send the PRODUCTION invoice (real email)"""
    print(f"\n📤 Sending PRODUCTION invoice: {invoice_id}")
    
    url = f"https://api-m.paypal.com/v2/invoicing/invoices/{invoice_id}/send"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {access_token}',
    }
    
    try:
        response = requests.post(url, headers=headers)
        print(f"Send response status: {response.status_code}")
        
        if response.status_code in [200, 202]:
            print("✅ PRODUCTION Invoice sent successfully!")
            print("📧 Real email sent to recipient!")
            return True
        else:
            print(f"❌ Failed to send production invoice: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to send production invoice: {e}")
        return False

def send_production_reminder(access_token, invoice_id):
    """Send PRODUCTION reminder (real email)"""
    print(f"\n📨 Sending PRODUCTION reminder for invoice: {invoice_id}")
    
    url = f"https://api-m.paypal.com/v2/invoicing/invoices/{invoice_id}/remind"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {access_token}',
    }
    
    reminder_data = {
        "subject": "Payment Reminder from Achintya's Voice Assistant",
        "note": "Hi! This is a payment reminder sent by my voice assistant. Please settle the outstanding balance when convenient. Thanks!"
    }
    
    try:
        response = requests.post(url, headers=headers, json=reminder_data)
        print(f"Reminder response status: {response.status_code}")
        
        if response.status_code == 204:
            print("✅ PRODUCTION Reminder sent successfully!")
            print("📧 Real reminder email sent!")
            return True
        else:
            print(f"❌ Failed to send production reminder: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to send production reminder: {e}")
        return False

def main():
    print("🚨 PayPal PRODUCTION Test (Real Emails)")
    print("=" * 50)
    print("⚠️  WARNING: This will send REAL emails!")
    print("⚠️  Only run this if you want to test real email delivery")
    print("=" * 50)
    
    confirm = input("Type 'YES' to continue with PRODUCTION test: ")
    if confirm != 'YES':
        print("❌ Test cancelled")
        return
    
    # Use international email for testing (PayPal blocks India)
    recipient_email = "test@example.com"  # International test email
    print(f"📧 Testing with international email: {recipient_email}")
    print("💡 PayPal blocks invoices to Indian customers")
    
    # Step 1: Get production token
    token = get_paypal_token_production()
    if not token:
        return
    
    # Step 2: Create invoice
    invoice_id = create_production_invoice(token, recipient_email)
    if not invoice_id:
        return
    
    # Step 3: Send invoice (this sends the email!)
    sent = send_production_invoice(token, invoice_id)
    if not sent:
        return
    
    # Step 4: Send reminder (this sends reminder email!)
    reminder_sent = send_production_reminder(token, invoice_id)
    
    if reminder_sent:
        print(f"\n🎉 COMPLETE SUCCESS!")
        print(f"📧 Real emails sent to {recipient_email}")
        print(f"📋 Invoice ID: {invoice_id}")
        print(f"✅ Invoice email sent")
        print(f"✅ Reminder email sent")
        print(f"\n🔍 Check your email inbox now!")
    else:
        print(f"\n⚠️  Invoice sent but reminder failed")
        print(f"📧 Check {recipient_email} for the invoice email")

if __name__ == "__main__":
    main()