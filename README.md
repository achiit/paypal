# Voice Assistant with Web Chat Interface

This application provides a voice assistant with both Twilio phone integration and a web-based chat interface.

## Features

- **Twilio Integration**: Handle incoming phone calls with voice-to-voice conversation
- **Web Chat Interface**: Direct text and voice chat through a web browser
- **Expense Management**: Integration with Splitwise for expense tracking and payments
- **Multi-language Support**: Supports multiple languages through SarvamAI

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your API keys:
   ```
   SARVAM_API_KEY=your_sarvam_api_key
   TOOLS_API_BASE_URL=your_tools_api_base_url
   SPLITWISE_API_KEY=your_splitwise_api_key
   CASHFREE_CLIENT_ID=your_cashfree_client_id
   CASHFREE_CLIENT_SECRET=your_cashfree_client_secret
   ```

3. Run the application:
   ```bash
   python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Usage

### Web Chat Interface

1. Open your browser and go to `http://localhost:8000`
2. You'll see a modern chat interface
3. You can:
   - Type messages and press Enter or click Send
   - Click the microphone button to record voice messages
   - View conversation history in the chat window

### Twilio Phone Integration

1. Configure your Twilio webhook to point to `/incoming_call`
2. Update the WebSocket URL in the code to your ngrok or public URL
3. Incoming calls will be handled automatically

### Available Commands

The assistant can help with:
- **Expense queries**: "Show me my recent expenses"
- **User information**: "What's my account details?"
- **Payments**: "Pay John 250 rupees"
- **General conversation**: Greetings, questions, etc.

## File Structure

```
├── main.py                 # Main FastAPI application
├── static/
│   ├── index.html         # Web chat interface
│   └── chat.js           # JavaScript for WebSocket communication
├── audio_logs/           # Incoming audio logs
├── outgoing_audio_logs/  # Outgoing audio logs
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## API Endpoints

- `GET /`: Serves the web chat interface
- `POST /incoming_call`: Twilio webhook for incoming calls
- `WebSocket /ws`: Twilio media stream endpoint
- `WebSocket /chat-ws`: Web chat interface endpoint

## Technical Details

- **Frontend**: Vanilla HTML/CSS/JavaScript with WebSocket
- **Backend**: FastAPI with WebSocket support
- **Audio Processing**: SarvamAI for speech-to-text and text-to-speech
- **Voice Format**: Handles both Twilio's mulaw and web audio formats
- **Real-time**: WebSocket-based real-time communication

## Troubleshooting

1. **WebSocket connection issues**: Check that the server is running and accessible
2. **Audio not working**: Ensure microphone permissions are granted in the browser
3. **API errors**: Verify all API keys are correctly set in the `.env` file
4. **Twilio integration**: Make sure ngrok is running and webhook URLs are updated