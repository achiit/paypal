#!/usr/bin/env python3
"""
Direct email test using Gmail SMTP (real emails)
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

load_dotenv()

def send_test_email():
    """Send a real email using Gmail SMTP"""
    
    # Gmail SMTP settings
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    
    # You'll need to set up Gmail App Password
    sender_email = "your-email@gmail.com"  # Replace with your Gmail
    sender_password = "your-app-password"   # Replace with Gmail App Password
    recipient_email = "achihsingh@gmail.com"
    
    print("📧 Testing direct email sending...")
    print(f"From: {sender_email}")
    print(f"To: {recipient_email}")
    
    # Create message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = "🤖 Payment Reminder from Voice Assistant"
    
    # Email body
    body = """
    Hi!
    
    This is a test payment reminder sent by your voice assistant.
    
    Outstanding Balance: ₹1,339.01 ($16.13)
    
    Please settle this amount when convenient.
    
    Thanks!
    Achintya's Voice Assistant
    """
    
    message.attach(MIMEText(body, "plain"))
    
    try:
        # Connect to Gmail SMTP
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Enable encryption
        server.login(sender_email, sender_password)
        
        # Send email
        text = message.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        
        print("✅ Email sent successfully!")
        print(f"📧 Check {recipient_email} for the payment reminder")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        print("\n💡 To use Gmail SMTP:")
        print("1. Enable 2-factor authentication on your Gmail")
        print("2. Generate an App Password: https://myaccount.google.com/apppasswords")
        print("3. Update sender_email and sender_password in this script")
        return False

def main():
    print("📧 Direct Email Test (Real Email)")
    print("=" * 40)
    
    success = send_test_email()
    
    if success:
        print("\n🎉 SUCCESS! Real email sent!")
        print("This proves email delivery works.")
        print("We can integrate this into the voice assistant.")
    else:
        print("\n❌ Email failed. Check Gmail setup.")

if __name__ == "__main__":
    main()