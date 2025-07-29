import os
import base64
import io
import logging
import wave
import audioop
import pywav
import requests
import json
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, File, UploadFile, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from twilio.twiml.voice_response import VoiceResponse, Connect
from sarvamai import SarvamAI
from dotenv import load_dotenv
from nnmnkwii.preprocessing import mulaw_quantize, inv_mulaw_quantize, mulaw, inv_mulaw
from datetime import datetime
from scipy.signal import resample
from tempfile import NamedTemporaryFile
import tempfile
# from scikits.audiolab import Sndfile

# --- Configuration ---
# Load environment variables from .env file
# Create a .env file in the same directory and add your keys
# SARVAM_API_KEY="your_sarvam_api_key"
# TWILIO_ACCOUNT_SID="your_twilio_account_sid"
# TWILIO_AUTH_TOKEN="your_twilio_auth_token"
load_dotenv()

# Get credentials from environment
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TOOLS_API_BASE_URL = os.getenv("TOOLS_API_BASE_URL")
SPLITWISE_API_KEY = os.getenv("SPLITWISE_API_KEY")
CASHFREE_CLIENT_ID = os.getenv("CASHFREE_CLIENT_ID")
CASHFREE_CLIENT_SECRET = os.getenv("CASHFREE_CLIENT_SECRET")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Voice Assistant API", version="1.0.0")

# Add CORS middleware for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the web interface
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the chat interface
@app.get("/")
async def get_chat_interface():
    """Serve the chat interface HTML page."""
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return Response(content=html_content, media_type="text/html")



# Initialize SarvamAI client
try:
    sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
    logger.info("SarvamAI client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize SarvamAI client: {e}")
    sarvam_client = None

# --- Twilio Webhook for Incoming Calls ---
@app.post("/incoming_call")
async def handle_incoming_call(request: Request):
    """
    Handles incoming calls from Twilio.
    Responds with TwiML to connect the call to our WebSocket stream.
    """
    logger.info("Incoming call received")
    
    # Log the incoming request for debugging
    try:
        form_data = await request.form()
        logger.info(f"Twilio request data: {dict(form_data)}")
    except Exception as e:
        logger.warning(f"Could not parse form data: {e}")
    
    twiml_response = VoiceResponse()
    
    # Add a brief greeting before connecting
    twiml_response.say("Hello! Connecting you to the voice assistant.", voice='alice')
    
    # The <Connect> verb will establish a media stream
    connect = Connect()
    # IMPORTANT: Replace with your actual ngrok URL
    connect.stream(url="wss://5fd4c229db5f.ngrok-free.app/ws")
    twiml_response.append(connect)
    
    logger.info("Responding with TwiML to connect to WebSocket.")
    logger.info(f"TwiML Response: {str(twiml_response)}")
    
    return Response(content=str(twiml_response), media_type="application/xml")

# --- WebSocket for Bidirectional Audio Streaming ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Handles the bidirectional audio stream with Twilio.
    """
    try:
        # Log connection details for debugging
        logger.info(f"🔌 WebSocket connection attempt from: {websocket.client}")
        logger.info(f"🔌 Headers: {dict(websocket.headers)}")
        
        await websocket.accept()
        logger.info("🔌 WebSocket connection established with Twilio.")
        
        # Send acknowledgment
        await websocket.send_json({"type": "connection_established"})
        
        audio_buffer = bytearray()
        stream_sid = None
        
        while True:
            message = await websocket.receive_json()
            event = message.get("event")

            if event == "start":
                stream_sid = message["start"]["streamSid"]
                logger.info(f"🎵 Twilio media stream started (SID: {stream_sid}).")
                
                # Send acknowledgment back to Twilio
                await websocket.send_json({
                    "event": "start_ack",
                    "streamSid": stream_sid
                })

            elif event == "media":
                payload = message["media"]["payload"]
                audio_data = base64.b64decode(payload)
                audio_buffer.extend(audio_data)

                # 8000 bytes = 1 second for 8-bit, 8000Hz, 1-channel audio
                if len(audio_buffer) > 24000: # Process after ~3 seconds of audio
                    logger.info(f"Buffer full ({len(audio_buffer)} bytes), processing audio...")
                    
                    # --- Start of Conversational Loop ---
                    
                    # 1. Prepare audio data for transcription
                    wav_bytes = convert_mulaw_to_wav_bytes(bytes(audio_buffer))
                    
                    if wav_bytes:
                        # 2. Transcribe audio to text
                        transcription = transcribe_audio(wav_bytes)
                        if transcription and transcription.transcript:
                            # CORRECTED: Get the detected language from the STT response using the correct attribute 'language_code'.
                            # We default to 'en-IN' if the language code is not available.
                            detected_language = getattr(transcription, 'language_code', 'en-IN')
                            logger.info(f"Detected language: {detected_language}")

                            # 3. Get a response from the LLM
                            logger.info(f"LLM INPUT (Transcription): {transcription.transcript}")
                            llm_response_text = get_llm_response(
                                transcription.transcript,
                                language_code=detected_language
                            )
                            
                            if llm_response_text:
                                logger.info(f"LLM OUPUT (Response): {llm_response_text}")
                                
                                # 4. Convert the LLM's text response to speech
                                # NEW: Pass the detected language to the TTS function.
                                response_audio_wav = convert_text_to_speech(
                                    llm_response_text,
                                    language_code=detected_language
                                )

                                if response_audio_wav:
                                    # --- Start of Comprehensive Outgoing Audio Logging ---
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                                    # Log the original, clean WAV from the TTS service
                                    tts_log_filename = f"outgoing_audio_logs/tts_output_{timestamp}.wav"
                                    with open(tts_log_filename, "wb") as log_file:
                                        log_file.write(response_audio_wav)
                                    logger.info(f"Saved original TTS audio to: {tts_log_filename}")

                                    # 5. Convert response audio to raw mulaw bytes for Twilio
                                    response_audio_mulaw = convert_wav_to_mulaw_bytes(response_audio_wav)
                                    
                                    if response_audio_mulaw:
                                        # Log the final raw mulaw bytestream being sent to Twilio
                                        mulaw_log_filename = f"outgoing_audio_logs/twilio_stream_{timestamp}.ulaw"
                                        with open(mulaw_log_filename, "wb") as log_file:
                                            log_file.write(response_audio_mulaw)
                                        logger.info(f"Saved final mulaw stream to: {mulaw_log_filename}")

                                        # 6. Send audio back to Twilio
                                        payload = base64.b64encode(response_audio_mulaw).decode("utf-8")
                                        
                                        # --- Start of Final Verification Log ---
                                        logger.info("Preparing to send media response to Twilio.")
                                        logger.info(f"  - Event: media")
                                        logger.info(f"  - Stream SID: {stream_sid}")
                                        logger.info(f"  - Payload Length (chars): {len(payload)}")
                                        # --- End of Final Verification Log ---
                                        
                                        await websocket.send_json({
                                            "event": "media",
                                            "streamSid": stream_sid,
                                            "media": {
                                                "payload": payload
                                            }
                                        })
                                        logger.info("Sent audio response back to Twilio.")

                    # --- End of Conversational Loop ---
                    
                    # Clear buffer after processing
                    audio_buffer.clear()

            elif event == "stop":
                logger.info("Twilio media stream stopped.")
                # Process any remaining audio in the buffer to catch the last words.
                if audio_buffer:
                    logger.info("Processing remaining audio in buffer on stop event.")
                    wav_bytes = convert_mulaw_to_wav_bytes(bytes(audio_buffer))
                    if wav_bytes:
                        transcription = transcribe_audio(wav_bytes)
                        if transcription and transcription.transcript:
                            # We'll just log the final transcription and not send a response,
                            # as the stream is closing.
                            logger.info(f"Final transcription: {transcription.transcript}")
                    audio_buffer.clear()
                break
                       
    except WebSocketDisconnect:
        logger.warning("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"Error in WebSocket: {e}", exc_info=True)
    finally:
        logger.info("Closing WebSocket connection.")

# --- WebSocket for Direct Chat Interface ---
@app.websocket("/chat-ws")
async def chat_websocket_endpoint(websocket: WebSocket):
    """
    Handles direct chat messages from the web interface.
    """
    await websocket.accept()
    logger.info("Chat WebSocket connection established.")
    
    try:
        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")
            
            if message_type == "text":
                # Handle text message
                user_message = message.get("message")
                logger.info(f"Received text message: {user_message}")
                
                # Get LLM response
                llm_response = get_llm_response(user_message)
                
                # Send text response
                await websocket.send_json({
                    "type": "response",
                    "message": llm_response
                })
                
                # Generate and send audio response
                audio_response = convert_text_to_speech(llm_response)
                if audio_response:
                    audio_base64 = base64.b64encode(audio_response).decode('utf-8')
                    await websocket.send_json({
                        "type": "audio_response",
                        "audio": audio_base64
                    })
                
            elif message_type == "audio":
                # Handle audio message
                audio_base64 = message.get("audio")
                logger.info("Received audio message")
                
                try:
                    # Decode base64 audio
                    audio_data = base64.b64decode(audio_base64)
                    
                    # For web audio, we need to handle different format than Twilio
                    # This is a simplified version - you might need to adjust based on the actual audio format
                    transcription = transcribe_web_audio(audio_data)
                    
                    if transcription and transcription.transcript:
                        # Send transcription back
                        await websocket.send_json({
                            "type": "transcription",
                            "message": transcription.transcript
                        })
                        
                        # Get LLM response
                        detected_language = getattr(transcription, 'language_code', 'en-IN')
                        llm_response = get_llm_response(transcription.transcript, language_code=detected_language)
                        
                        # Send text response
                        await websocket.send_json({
                            "type": "response",
                            "message": llm_response
                        })
                        
                        # Generate and send audio response
                        audio_response = convert_text_to_speech(llm_response, language_code=detected_language)
                        if audio_response:
                            audio_base64 = base64.b64encode(audio_response).decode('utf-8')
                            await websocket.send_json({
                                "type": "audio_response",
                                "audio": audio_base64
                            })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Could not transcribe audio"
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": "Error processing audio"
                    })
                    
    except WebSocketDisconnect:
        logger.warning("Chat WebSocket disconnected.")
    except Exception as e:
        logger.error(f"Error in chat WebSocket: {e}", exc_info=True)
    finally:
        logger.info("Closing chat WebSocket connection.")

# --- Pydantic Models for Flutter API ---
class TextMessageRequest(BaseModel):
    message: str
    language_code: str = "en-IN"

class TextMessageResponse(BaseModel):
    success: bool
    message: str
    audio_base64: str = None
    error: str = None

class AudioTranscriptionResponse(BaseModel):
    success: bool
    transcription: str = None
    message: str = None
    audio_base64: str = None
    error: str = None

# --- REST API Endpoints for Flutter ---

@app.post("/api/chat/text", response_model=TextMessageResponse)
async def chat_text_endpoint(request: TextMessageRequest):
    """
    Handle text messages from Flutter app
    Returns both text response and audio (base64)
    """
    try:
        logger.info(f"Flutter text request: {request.message}")
        
        # Get LLM response
        llm_response = get_llm_response(request.message, request.language_code)
        
        # Generate audio response
        audio_response = convert_text_to_speech(llm_response, request.language_code)
        audio_base64 = None
        if audio_response:
            audio_base64 = base64.b64encode(audio_response).decode('utf-8')
        
        return TextMessageResponse(
            success=True,
            message=llm_response,
            audio_base64=audio_base64
        )
        
    except Exception as e:
        logger.error(f"Error in text chat: {e}")
        return TextMessageResponse(
            success=False,
            error=str(e)
        )

@app.post("/api/chat/audio", response_model=AudioTranscriptionResponse)
async def chat_audio_endpoint(audio_file: UploadFile = File(...), language_code: str = "en-IN"):
    """
    Handle audio messages from Flutter app
    Returns transcription, text response, and audio response (base64)
    """
    try:
        logger.info(f"Flutter audio request received")
        
        # Read audio file
        audio_data = await audio_file.read()
        
        # Transcribe audio
        transcription = transcribe_web_audio(audio_data)
        
        if not transcription or not transcription.transcript:
            return AudioTranscriptionResponse(
                success=False,
                error="Could not transcribe audio"
            )
        
        detected_language = getattr(transcription, 'language_code', language_code)
        logger.info(f"Transcribed: {transcription.transcript}")
        
        # Get LLM response
        llm_response = get_llm_response(transcription.transcript, detected_language)
        
        # Generate audio response
        audio_response = convert_text_to_speech(llm_response, detected_language)
        audio_base64 = None
        if audio_response:
            audio_base64 = base64.b64encode(audio_response).decode('utf-8')
        
        return AudioTranscriptionResponse(
            success=True,
            transcription=transcription.transcript,
            message=llm_response,
            audio_base64=audio_base64
        )
        
    except Exception as e:
        logger.error(f"Error in audio chat: {e}")
        return AudioTranscriptionResponse(
            success=False,
            error=str(e)
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint for Flutter app"""
    return {"status": "healthy", "message": "Voice Assistant API is running"}

@app.get("/test_webhook")
async def test_webhook():
    """Test endpoint to verify webhook is working"""
    twiml_response = VoiceResponse()
    twiml_response.say("Webhook is working correctly!", voice='alice')
    return Response(content=str(twiml_response), media_type="application/xml")

@app.get("/api/user/profile")
async def get_user_profile():
    """Get current user profile"""
    try:
        user_data = _get_current_user_identity()
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "success": True,
            "user": {
                "name": f"{user_data.get('first_name', '')} {user_data.get('last_name', '')}".strip(),
                "email": user_data.get('email', ''),
                "id": user_data.get('id', ''),
                "currency": user_data.get('default_currency', 'USD')
            }
        }
    except Exception as e:
        logger.error(f"Error getting user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/expenses")
async def get_expenses_api():
    """Get expenses and balances"""
    try:
        result = call_tool("get_expenses", {})
        return {
            "success": True,
            "data": json.loads(result)
        }
    except Exception as e:
        logger.error(f"Error getting expenses: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Audio Conversion Utilities ---

def convert_mulaw_to_wav_bytes(mulaw_bytes: bytes) -> bytes:
    """
    Packages raw mulaw bytes from Twilio into a WAV file container.
    This does NOT decode the audio, it just puts the raw bytes in a recognizable format.
    """
    try:
        # pywav needs to write to a real file, so we use a temporary file.
        with NamedTemporaryFile(suffix=".wav", delete=True) as tmpfile:
            wave_write = pywav.WavWrite(tmpfile.name, 1, 8000, 8, 7)  # 7 = µ-law encoding
            wave_write.write(mulaw_bytes)
            wave_write.close()
            
            # Read the bytes from the temporary file we just created
            tmpfile.seek(0)
            wav_bytes = tmpfile.read()
        
        return wav_bytes
    except Exception as e:
        logger.error(f"Failed to convert mulaw to wav: {e}", exc_info=True)
        return None

def convert_wav_to_mulaw_bytes(wav_bytes: bytes) -> bytes:
    """
    Converts a standard 16-bit PCM WAV file into raw, headerless 8kHz µ-law
    bytes suitable for the Twilio media stream, using the standard audioop library.
    """
    try:
        # 1. Read the raw PCM audio frames from the WAV file bytes
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
            # Ensure audio is 16-bit mono PCM, which is what lin2ulaw expects.
            if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
                logger.error(
                    f"Unsupported WAV format: "
                    f"Sample width {wf.getsampwidth()}, channels {wf.getnchannels()}. "
                    f"Expected 16-bit mono."
                )
                return None
            
            # The TTS service should already provide 8kHz, but we log a warning if not.
            if wf.getframerate() != 8000:
                logger.warning(f"WAV sample rate is {wf.getframerate()}, not 8000Hz.")

            pcm_frames = wf.readframes(wf.getnframes())

        # 2. Convert the 16-bit linear PCM data to 8-bit µ-law.
        # The '2' indicates the sample width of the input data is 2 bytes (16-bit).
        mulaw_bytes = audioop.lin2ulaw(pcm_frames, 2)
        
        return mulaw_bytes

    except Exception as e:
        logger.error(f"Failed to convert wav to mulaw: {e}", exc_info=True)
        return None

def transcribe_web_audio(audio_data: bytes):
    """
    Transcribe audio from web interface using SarvamAI.
    Web audio is typically in different format than Twilio's mulaw.
    """
    if not sarvam_client:
        logger.error("SarvamAI client not available.")
        return None

    logger.info("Transcribing web audio with SarvamAI.")
    try:
        # Create a file-like object from the audio data
        audio_file_like = io.BytesIO(audio_data)
        audio_file_like.name = "web_audio.wav"
        
        # Use the speech-to-text API
        response = sarvam_client.speech_to_text.translate(
            file=audio_file_like,
            model="saaras:v2.5"
        )
        logger.info(f"Web audio transcription: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Web audio transcription failed: {e}")
        return None

# --- Test function for debugging ---
def test_expense_calculation():
    """Test function to verify expense calculation logic"""
    try:
        result = call_tool("get_expenses", {})
        logger.info(f"Test result: {result}")
        return result
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return None

# --- Tool Definitions and Execution ---

def summarize_expenses(expenses: list, limit: int = 15) -> list:
    """
    Summarizes a list of expenses based on the new API format, returning
    the key details for the most recent transactions, including emails.
    """
    summary = []
    # Process the most recent expenses up to the limit
    for expense in expenses[:limit]:
        summary.append({
            "description": expense.get('description'),
            "amount": expense.get('amount'),
            "currency": expense.get('currency_code'),
            "date": expense.get('date', '').split('T')[0],
            "from_user": expense.get('from'),
            "from_email": expense.get('from_email'),
            "to_user": expense.get('to'),
            "to_email": expense.get('to_email'),
            "settled": expense.get('settled')
        })
    return summary

# Define the schema for the tools the LLM can use.
# This tells the model what functions are available, what they do, and what parameters they take.
TOOLS = [
    {
        "name": "get_current_user",
        "description": "Fetches the details of the currently authenticated user from Splitwise. Use this to find out who the user is, their name, or their user ID.",
        "parameters": []
    },
    {
        "name": "get_expenses",
        "description": "Fetches a list of recent expenses. Use this when the user asks about their recent transactions, bills, or what they've spent money on.",
        "parameters": []
    },
    {
        "name": "initiate_payment",
        "description": "Starts the process of paying an outstanding expense to a specific person. Use this when the user wants to settle a debt or pay someone.",
        "parameters": [
            {"name": "recipient_name", "type": "string", "description": "The name of the person to pay."}
        ]
    }
]

def _get_current_user_identity() -> dict:
    """Internal helper to fetch the current user's details directly from Splitwise."""
    logger.info("Fetching current user identity from Splitwise...")
    url = "https://secure.splitwise.com/api/v3.0/get_current_user"
    headers = {
        'Authorization': f'Bearer {SPLITWISE_API_KEY}',
        'Content-Type': 'application/json'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        user_data = response.json()
        logger.info(f"Successfully fetched user: {user_data.get('user', {}).get('first_name', 'Unknown')}")
        return user_data.get('user', {})
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch current user identity: {e}")
        return {}

def call_tool(tool_name: str, parameters: dict):
    """
    Executes the appropriate API call based on the tool name provided by the LLM.
    """
    if tool_name == "get_current_user":
        logger.info("Executing tool: get_current_user")
        user_identity = _get_current_user_identity()
        if not user_identity:
            return json.dumps({"error": "Could not retrieve current user's identity."})
        
        # Return user data in a simple format for the LLM
        user_info = {
            "name": f"{user_identity.get('first_name', '')} {user_identity.get('last_name', '')}".strip(),
            "email": user_identity.get('email', ''),
            "id": user_identity.get('id', ''),
            "registration_status": user_identity.get('registration_status', ''),
            "default_currency": user_identity.get('default_currency', '')
        }
        logger.info(f"Tool 'get_current_user' returned user: {user_info['name']}")
        return json.dumps(user_info)
    
    elif tool_name == "get_expenses":
        logger.info("Executing tool: get_expenses - calling Splitwise directly")
        url = "https://secure.splitwise.com/api/v3.0/get_expenses"
        headers = {
            'Authorization': f'Bearer {SPLITWISE_API_KEY}',
            'Content-Type': 'application/json'
        }
        try:
            response = requests.get(f"{url}?limit=20", headers=headers)
            response.raise_for_status()
            expenses_data = response.json()
            
            expenses_list = expenses_data.get('expenses', [])
            logger.info(f"Successfully fetched {len(expenses_list)} expenses from Splitwise")
            
            # Get current user info for balance calculations
            current_user = _get_current_user_identity()
            current_user_id = current_user.get('id') if current_user else None
            logger.info(f"Current user ID: {current_user_id}")
            
            # Calculate net balances with each person
            person_balances = {}  # person_name -> net_amount (positive = they owe you, negative = you owe them)
            expense_details = []
            
            for expense in expenses_list:
                if expense.get('payment', False):  # Skip payment entries
                    continue
                    
                users = expense.get('users', [])
                current_user_balance = 0
                other_users = []
                
                # Find current user's balance and other users in this expense
                for user in users:
                    user_info = user.get('user', {})
                    user_id = user_info.get('id')
                    user_name = f"{user_info.get('first_name', '')} {user_info.get('last_name', '')}".strip()
                    
                    # net_balance in Splitwise: positive = they get money, negative = they owe money
                    net_balance = float(user.get('net_balance', 0))
                    
                    if user_id == current_user_id:
                        current_user_balance = net_balance
                    else:
                        other_users.append({
                            'name': user_name,
                            'balance': net_balance
                        })
                
                # If current user has negative balance, they owe money
                # If others have negative balance, they owe money
                if current_user_balance < 0:  # Current user owes money
                    for other_user in other_users:
                        if other_user['balance'] > 0:  # This person is owed money
                            person_name = other_user['name']
                            amount = abs(current_user_balance)  # Amount current user owes
                            
                            if person_name not in person_balances:
                                person_balances[person_name] = 0
                            person_balances[person_name] -= amount  # Negative = you owe them
                            
                elif current_user_balance > 0:  # Current user is owed money
                    for other_user in other_users:
                        if other_user['balance'] < 0:  # This person owes money
                            person_name = other_user['name']
                            amount = abs(other_user['balance'])  # Amount they owe
                            
                            if person_name not in person_balances:
                                person_balances[person_name] = 0
                            person_balances[person_name] += amount  # Positive = they owe you
                
                # Add expense details
                expense_details.append({
                    "description": expense.get('description', ''),
                    "amount": expense.get('cost', '0'),
                    "date": expense.get('date', '').split('T')[0] if expense.get('date') else '',
                    "your_balance": current_user_balance,
                    "currency": expense.get('currency_code', 'INR')
                })
            
            # Format the final result
            balance_summary = []
            for person, net_amount in person_balances.items():
                if abs(net_amount) > 0.01:  # Ignore very small amounts
                    balance_summary.append({
                        "person": person,
                        "amount": abs(net_amount),
                        "direction": "they_owe_you" if net_amount > 0 else "you_owe_them",
                        "amount_formatted": f"₹{abs(net_amount):.2f}"
                    })
            
            result = {
                "balances": balance_summary,
                "recent_expenses": expense_details[:10],
                "total_people": len(balance_summary)
            }
            
            logger.info(f"Calculated balances: {balance_summary}")
            return json.dumps(result)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Splitwise API call failed: {e}")
            return json.dumps({"error": f"Failed to fetch expenses from Splitwise: {str(e)}"})

    elif tool_name == "initiate_payment":
        logger.info(f"--- Starting Intelligent Payment Flow for: {parameters.get('recipient_name')} ---")
        recipient_name_query = parameters.get("recipient_name")
        if not recipient_name_query:
            return json.dumps({"error": "I need to know who you want to pay. Please provide a name."})

        # Step 1: Establish self-identity. Who am I?
        current_user = _get_current_user_identity()
        if not current_user:
            return json.dumps({"error": "I couldn't identify who you are, so I can't make a payment."})
        current_user_name = f"{current_user.get('first_name', '')} {current_user.get('last_name', '')}".strip()
        logger.info(f"Step 1: Identity confirmed as '{current_user_name}'.")

        # Step 2: Get all expenses for context directly from Splitwise.
        logger.info("Step 2: Fetching all expenses from Splitwise to calculate net balance.")
        expenses_url = "https://secure.splitwise.com/api/v3.0/get_expenses"
        expenses_headers = {
            'Authorization': f'Bearer {SPLITWISE_API_KEY}',
            'Content-Type': 'application/json'
        }
        try:
            expenses_response = requests.get(f"{expenses_url}?limit=100", headers=expenses_headers)
            expenses_response.raise_for_status()
            all_expenses = expenses_response.json().get('expenses', [])
            logger.info(f"Successfully fetched {len(all_expenses)} expense records from Splitwise.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Splitwise API call failed: {e}")
            return json.dumps({"error": "I couldn't retrieve the list of expenses to find the payment details."})

        # Step 3: Calculate the net balance between the current user and the recipient.
        logger.info(f"Step 3: Calculating net balance between '{current_user_name}' and '{recipient_name_query}'.")
        net_balance = 0.0
        recipient_email = None
        recipient_full_name = None

        current_user_id = current_user.get('id')
        recipient_query_words = set(recipient_name_query.lower().split())

        for expense in all_expenses:
            if expense.get('payment', False):  # Skip settled expenses
                continue

            expense_cost = float(expense.get('cost', 0.0))
            users = expense.get('users', [])
            
            current_user_share = 0
            recipient_user_info = None
            
            for user in users:
                user_info = user.get('user', {})
                user_id = user_info.get('id')
                owed_share = float(user.get('owed_share', 0))
                paid_share = float(user.get('paid_share', 0))
                
                if user_id == current_user_id:
                    # This is the current user's share
                    current_user_share = owed_share - paid_share
                else:
                    # Check if this might be the recipient
                    user_name = f"{user_info.get('first_name', '')} {user_info.get('last_name', '')}".strip()
                    user_name_words = set(user_name.lower().split())
                    
                    if recipient_query_words.issubset(user_name_words):
                        recipient_user_info = user_info
                        recipient_share = owed_share - paid_share
                        
                        # If current user owes money in this expense
                        if current_user_share > 0:
                            net_balance += current_user_share
                            if not recipient_email:
                                recipient_email = user_info.get('email')
                                recipient_full_name = user_name
        
        logger.info(f"Final calculated net balance is: {net_balance:.2f}")

        # Step 4: Act based on the calculated net balance.
        if net_balance <= 0:
            message = f"There is no outstanding balance for you to pay to {recipient_name_query}. "
            if net_balance < 0:
                message += f"In fact, they owe you {abs(net_balance):.2f}."
            else:
                message += "Your balance appears to be settled."
            return json.dumps({"error": message})

        if not recipient_email:
            return json.dumps({"error": f"I calculated that you owe {net_balance:.2f}, but I couldn't find an email for {recipient_name_query} to send the payment."})

        # Step 5: If a payment is needed, call the payment API with the exact payload.
        payment_payload = {
            "customer_email": recipient_email,
            "link_amount": int(net_balance * 100),
            "customer_name": recipient_full_name or recipient_name_query
        }
        
        logger.info(f"Step 4: Creating payment link directly with Cashfree")
        
        # Create payment link directly with Cashfree
        import uuid
        payment_url = "https://sandbox.cashfree.com/pg/links"
        payment_headers = {
            'Content-Type': 'application/json',
            'x-api-version': '2023-08-01',
            'x-client-id': CASHFREE_CLIENT_ID,
            'x-client-secret': CASHFREE_CLIENT_SECRET
        }
        
        cashfree_payload = {
            "link_id": f"link_{uuid.uuid4().hex[:8]}",
            "link_amount": int(net_balance * 100),  # Convert to paise
            "link_currency": "INR",
            "link_purpose": f"Payment to {recipient_full_name or recipient_name_query}",
            "customer_details": {
                "customer_name": recipient_full_name or recipient_name_query,
                "customer_email": recipient_email,
                "customer_phone": "9999999999"  # Default phone
            },
            "link_notify": {
                "send_sms": False,
                "send_email": True
            }
        }

        try:
            payment_response = requests.post(payment_url, headers=payment_headers, json=cashfree_payload)
            payment_response.raise_for_status()
            payment_data = payment_response.json()
            logger.info(f"Payment link created successfully: {payment_data.get('link_url', 'No URL')}")
            
            return json.dumps({
                "success": True,
                "amount": net_balance,
                "recipient": recipient_full_name or recipient_name_query,
                "payment_link": payment_data.get('link_url', ''),
                "message": f"Payment link created for ₹{net_balance:.2f} to {recipient_full_name or recipient_name_query}"
            })
        except requests.exceptions.RequestException as e:
            logger.error(f"Cashfree payment link creation failed: {e}")
            return json.dumps({"error": "I tried to create the payment link, but the request to the payment service failed."})
    else:
        logger.warning(f"LLM tried to call an unknown tool: {tool_name}")
        return json.dumps({"error": "Unknown tool."})

# --- SarvamAI Speech-to-Text Function (adapted from your script) ---
def transcribe_audio(audio_bytes: bytes):
    """
    Transcribe audio using SarvamAI's speech translation API.
    Note: Twilio sends audio in mulaw format. SarvamAI might need a different
    format like WAV. We may need to add a conversion step here.
    """
    if not sarvam_client:
        logger.error("SarvamAI client not available.")
        return None

    logger.info("Sending audio to SarvamAI for transcription.")
    try:
        # --- Start of Debugging Block ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_filename = f"audio_logs/transcription_input_{timestamp}.wav"
        
        # Use the pywav writer for high-fidelity logging
        wave_write = pywav.WavWrite(log_filename, 1, 8000, 8, 7) # 7 = µ-law
        wave_write.write(audio_bytes)
        wave_write.close()

        logger.info(f"Saved audio for debugging to: {log_filename}")
        # --- End of Debugging Block ---

        # The API needs a file-like object.
        # We now pass the properly containerized WAV bytes
        audio_file_like = io.BytesIO(convert_mulaw_to_wav_bytes(audio_bytes))
        # We now have a WAV file, so we name it accordingly.
        audio_file_like.name = "audio.wav" 

        # IMPORTANT: This is the speech-to-text model.
        response = sarvam_client.speech_to_text.translate(
            file=audio_file_like,
            model="saaras:v2.5" 
        )
        logger.info(f"Received transcription: {response}")
        return response
            
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return None

# --- SarvamAI Language Model (LLM) Function ---
def normalize_query(text: str) -> str:
    """
    Normalize user query to handle voice transcription variations
    """
    text = text.lower().strip()
    
    # Common voice transcription fixes
    replacements = {
        'tell me your profile details': 'tell me your profile details',
        'tell me profile details': 'tell me your profile details',
        'show me profile': 'tell me your profile details',
        'my profile': 'tell me your profile details',
        'profile details': 'tell me your profile details',
        'account details': 'tell me your profile details',
        'user details': 'tell me your profile details',
        
        'how much money do i have to take back': 'show my expenses',
        'how much money do i have to take': 'show my expenses',
        'how much do i have to take': 'show my expenses',
        'money to take': 'show my expenses',
        'who owes me': 'show my expenses',
        'what do i owe': 'show my expenses',
        'show expenses': 'show my expenses',
        'my expenses': 'show my expenses',
        'recent expenses': 'show my expenses',
        'balances': 'show my expenses',
        'outstanding': 'show my expenses',
        
        'send payment reminder': 'show my expenses',
        'payment reminder': 'show my expenses',
        'remind payment': 'show my expenses',
    }
    
    # Check for exact matches first
    for pattern, replacement in replacements.items():
        if pattern in text:
            return replacement
    
    return text

def get_llm_response(text: str, language_code: str = "en-IN"):
    """
    Manages the interaction with the LLM, including tool-calling logic.
    """
    if not sarvam_client:
        logger.error("SarvamAI client not available.")
        return "The AI model is currently unavailable. Please try again later."

    # Normalize the input query for better consistency
    normalized_text = normalize_query(text)
    logger.info(f"Original: '{text}' -> Normalized: '{normalized_text}'")

    # 1. First Pass: Tool Selection with more explicit rules
    system_prompt_for_tool_selection = f"""
You are a financial assistant. Analyze the user query and respond with the appropriate tool or conversation.

STRICT RULES:
1. If query contains ANY of these words/phrases → use get_expenses tool:
   - "expenses", "money", "owe", "take", "balances", "outstanding", "payment", "reminder"
   - "how much", "show", "recent", "transactions", "bills"

2. If query asks about "profile", "details", "account", "user" → use get_current_user tool

3. If query is greeting/casual → respond conversationally

4. For tool usage, respond ONLY with JSON: {{"tool_name": "tool_name", "parameters": {{}}}}

5. For conversation, respond naturally in {language_code}

Query: "{normalized_text}"

Available tools: {json.dumps(TOOLS, indent=2)}
"""
    
    # Pre-determine tool based on keywords (more reliable than LLM for voice)
    def determine_tool_by_keywords(query: str) -> dict:
        query_lower = query.lower()
        
        # Profile/user related keywords
        profile_keywords = ['profile', 'details', 'account', 'user', 'my details', 'who am i']
        if any(keyword in query_lower for keyword in profile_keywords):
            return {"tool_name": "get_current_user", "parameters": {}}
        
        # Expense/money related keywords
        expense_keywords = ['money', 'owe', 'take', 'expenses', 'balances', 'outstanding', 
                           'payment', 'reminder', 'how much', 'show', 'recent', 'transactions', 
                           'bills', 'aditya', 'nishant', 'omkar', 'dainik']
        if any(keyword in query_lower for keyword in expense_keywords):
            return {"tool_name": "get_expenses", "parameters": {}}
        
        # Greeting keywords
        greeting_keywords = ['hi', 'hello', 'hey', 'good morning', 'good evening', 'how are you']
        if any(keyword in query_lower for keyword in greeting_keywords):
            return None  # No tool needed
        
        # Default to expenses for ambiguous queries
        return {"tool_name": "get_expenses", "parameters": {}}
    
    # Try keyword-based tool determination first (more reliable for voice)
    keyword_tool = determine_tool_by_keywords(normalized_text)
    
    if keyword_tool:
        logger.info(f"Tool determined by keywords: {keyword_tool}")
        tool_name = keyword_tool.get("tool_name")
        
        if tool_name:
            # Execute the tool directly
            logger.info(f"Executing tool: {tool_name} with parameters: {keyword_tool.get('parameters', {})}")
            tool_result = call_tool(tool_name, keyword_tool.get("parameters", {}))
            logger.info(f"Tool execution completed. Result length: {len(str(tool_result))}")
            
            # Check if tool result is valid
            if not tool_result:
                logger.error("Tool returned empty result")
                return "I couldn't retrieve the information you requested. Please try again."
            
            # Generate final response using the tool result
            system_prompt_for_final_response = f"""You are a financial assistant. Give direct, clear answers based on the expense data provided.

LANGUAGE: Respond in {language_code}.

RULES:
1. When user asks "How much money do I have to take from [person]" → Show money THEY owe YOU (direction: "they_owe_you")
2. When user asks "What do I owe [person]" → Show money YOU owe THEM (direction: "you_owe_them")
3. When user asks "Send payment reminder to [person]" → Show money YOU owe THEM (direction: "you_owe_them")
4. Be specific with amounts and currency
5. Use plain text, no formatting

EXAMPLES:
- If data shows Aditya with direction "they_owe_you" and amount 50: "Aditya owes you 50 rupees"
- If data shows John with direction "you_owe_them" and amount 100: "You owe John 100 rupees"
- If no balance found: "There are no outstanding balances with [person]"
- For payment reminders: Focus on what YOU owe THEM

Answer directly based on the balance data provided."""
            
            final_messages = [
                {"role": "system", "content": system_prompt_for_final_response},
                {"role": "user", "content": f"My original question was: '{text}'"},
                {"role": "assistant", "content": f"I have run the tool '{tool_name}' and the result is: {tool_result}"},
                {"role": "user", "content": "Now, please give me the final answer based on this information."}
            ]
            
            logger.info(f"Sending tool result to LLM for final response generation.")
            final_response = sarvam_client.chat.completions(
                messages=final_messages,
                max_tokens=300,
                temperature=0.7,
            )
            final_content = final_response.choices[0].message.content
            logger.info(f"Received from LLM (final response): {final_content}")
            return final_content
    
    # Fallback to LLM-based tool selection for complex queries
    messages = [
        {"role": "system", "content": system_prompt_for_tool_selection},
        {"role": "user", "content": normalized_text}
    ]
    
    logger.info(f"Using LLM for tool selection: {normalized_text}")
    try:
        response = sarvam_client.chat.completions(
            messages=messages,
            max_tokens=550,
            temperature=0.0,
        )
        llm_output = response.choices[0].message.content.strip()
        logger.info(f"Received from LLM (initial pass): {llm_output}")

        # Handle empty or whitespace-only responses
        if not llm_output:
            logger.warning("LLM returned empty response, providing default greeting")
            return "Hello! I'm your financial assistant. How can I help you today?"

        # 2. Check if the LLM wants to call a tool
        try:
            tool_call_request = json.loads(llm_output)
            tool_name = tool_call_request.get("tool_name")
            
            if tool_name:
                # 3. Execute the tool
                logger.info(f"Executing tool: {tool_name} with parameters: {tool_call_request.get('parameters', {})}")
                tool_result = call_tool(tool_name, tool_call_request.get("parameters", {}))
                logger.info(f"Tool execution completed. Result length: {len(str(tool_result))}")
                
                # Check if tool result is valid
                if not tool_result:
                    logger.error("Tool returned empty result")
                    return "I couldn't retrieve the information you requested. Please try again."
                
                # 4. Second Pass: Generate Final Response
                # Now we send the tool's result back to the LLM to generate a human-friendly response.
                system_prompt_for_final_response = f"""You are a financial assistant. Give direct, clear answers based on the expense data provided.

LANGUAGE: Respond in {language_code}.

RULES:
1. When user asks "How much money do I have to take from [person]" → Show money THEY owe YOU (direction: "they_owe_you")
2. When user asks "What do I owe [person]" → Show money YOU owe THEM (direction: "you_owe_them")
3. When user asks "Send payment reminder to [person]" → Show money YOU owe THEM (direction: "you_owe_them")
4. Be specific with amounts and currency
5. Use plain text, no formatting

EXAMPLES:
- If data shows Aditya with direction "they_owe_you" and amount 50: "Aditya owes you 50 rupees"
- If data shows John with direction "you_owe_them" and amount 100: "You owe John 100 rupees"
- If no balance found: "There are no outstanding balances with [person]"
- For payment reminders: Focus on what YOU owe THEM

Answer directly based on the balance data provided."""
                
                final_messages = [
                    {"role": "system", "content": system_prompt_for_final_response},
                    {"role": "user", "content": f"My original question was: '{text}'"},
                    {"role": "assistant", "content": f"I have run the tool '{tool_name}' and the result is: {tool_result}"},
                    {"role": "user", "content": "Now, please give me the final answer based on this information."}
                ]
                
                logger.info(f"Sending tool result to LLM for final response generation.")
                final_response = sarvam_client.chat.completions(
                    messages=final_messages,
                    max_tokens=300, # Increased from 100 to allow for a full, detailed response
                    temperature=0.7,
                )
                logger.info(f"Final response: {final_response}")
                final_content = final_response.choices[0].message.content
                logger.info(f"Received from LLM (final response): {final_content}")
                return final_content
            else:
                # If it's valid JSON but not a tool call, treat as conversational
                return llm_output

        except (json.JSONDecodeError, AttributeError):
            # If the output is not a JSON object, it's a direct conversational response.
            logger.info("LLM response is conversational, not a tool call.")
            # Handle edge cases where response might be just "{}" or similar
            if llm_output.strip() in ["{}", "[]", ""]:
                return "Hello! I'm your financial assistant. How can I help you today?"
            return llm_output

    except Exception as e:
        logger.error(f"LLM request failed: {e}", exc_info=True)
        return "I'm sorry, I had trouble processing your request."

# --- SarvamAI Text-to-Speech (TTS) Function ---
def convert_text_to_speech(text: str, language_code: str = "en-IN"):
    """
    Converts text to speech using SarvamAI, correctly combines all audio chunks, 
    and returns a single, valid WAV audio byte string.
    """
    if not sarvam_client:
        logger.error("SarvamAI client not available.")
        return None
    
    logger.info(f"Sending to TTS: '{text}' in language: {language_code}")
    try:
        response = sarvam_client.text_to_speech.convert(
            text=text,
            target_language_code=language_code,
            speaker="anushka",
            model="bulbul:v2",
            speech_sample_rate=8000
        )
        
        audio_chunks_base64 = response.audios
        if not audio_chunks_base64:
            logger.error("TTS response contained no audio chunks.")
            return None

        logger.info(f"Received {len(audio_chunks_base64)} audio chunks from TTS. Combining them...")

        # Decode all chunks from base64 into a list of bytes
        decoded_chunks = [base64.b64decode(chunk) for chunk in audio_chunks_base64]

        # 1. Read the audio parameters from the first chunk.
        with wave.open(io.BytesIO(decoded_chunks[0]), 'rb') as wf:
            params = wf.getparams()

        # 2. Read the raw audio data (frames) from *all* chunks.
        all_frames = []
        for chunk_bytes in decoded_chunks:
            with wave.open(io.BytesIO(chunk_bytes), 'rb') as wf:
                all_frames.append(wf.readframes(wf.getnframes()))

        # 3. Create a new, final WAV file in memory.
        final_wav_buffer = io.BytesIO()
        with wave.open(final_wav_buffer, 'wb') as final_wf:
            # 4. Write the correct header and the combined audio data.
            final_wf.setparams(params)
            final_wf.writeframes(b"".join(all_frames))

        final_wav_bytes = final_wav_buffer.getvalue()
        logger.info("Successfully combined audio chunks into a single WAV file.")
        return final_wav_bytes

    except Exception as e:
        logger.error(f"TTS request or audio combination failed: {e}", exc_info=True)
        return None

# --- Main execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server.")
    # To run this app:
    # 1. Make sure you have a .env file with your credentials.
    # 2. In your terminal, run: uvicorn main:app --reload
    # 3. Use ngrok to expose your local port 8000 to the web.
    uvicorn.run(app, host="0.0.0.0", port=8000) 