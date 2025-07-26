# Flutter Integration Guide - Voice Assistant API

## 1. Dependencies (pubspec.yaml)

```yaml
dependencies:
  dio: ^5.3.2
  audioplayers: ^5.2.1
  record: ^5.0.4
  permission_handler: ^11.0.1
  path_provider: ^2.1.1
```

## 2. API Service Class

```dart
import 'package:dio/dio.dart';
import 'dart:convert';
import 'dart:io';

class VoiceAssistantService {
  static const String baseUrl = 'http://your-server-ip:8000'; // Replace with your server IP
  late Dio _dio;

  VoiceAssistantService() {
    _dio = Dio(BaseOptions(
      baseUrl: baseUrl,
      connectTimeout: const Duration(seconds: 30),
      receiveTimeout: const Duration(seconds: 30),
    ));
  }

  // Text Chat API
  Future<TextChatResponse> sendTextMessage(String message, {String languageCode = 'en-IN'}) async {
    try {
      final response = await _dio.post(
        '/api/chat/text',
        data: {
          'message': message,
          'language_code': languageCode,
        },
      );

      return TextChatResponse.fromJson(response.data);
    } on DioException catch (e) {
      throw Exception('Failed to send text message: ${e.message}');
    }
  }

  // Audio Chat API
  Future<AudioChatResponse> sendAudioMessage(File audioFile, {String languageCode = 'en-IN'}) async {
    try {
      FormData formData = FormData.fromMap({
        'audio_file': await MultipartFile.fromFile(
          audioFile.path,
          filename: 'audio.wav',
        ),
        'language_code': languageCode,
      });

      final response = await _dio.post(
        '/api/chat/audio',
        data: formData,
      );

      return AudioChatResponse.fromJson(response.data);
    } on DioException catch (e) {
      throw Exception('Failed to send audio message: ${e.message}');
    }
  }

  // Get User Profile
  Future<UserProfile> getUserProfile() async {
    try {
      final response = await _dio.get('/api/user/profile');
      return UserProfile.fromJson(response.data['user']);
    } on DioException catch (e) {
      throw Exception('Failed to get user profile: ${e.message}');
    }
  }

  // Get Expenses
  Future<ExpensesData> getExpenses() async {
    try {
      final response = await _dio.get('/api/expenses');
      return ExpensesData.fromJson(response.data['data']);
    } on DioException catch (e) {
      throw Exception('Failed to get expenses: ${e.message}');
    }
  }

  // Health Check
  Future<bool> healthCheck() async {
    try {
      final response = await _dio.get('/api/health');
      return response.data['status'] == 'healthy';
    } catch (e) {
      return false;
    }
  }
}
```

## 3. Data Models

```dart
// Text Chat Response Model
class TextChatResponse {
  final bool success;
  final String? message;
  final String? audioBase64;
  final String? error;

  TextChatResponse({
    required this.success,
    this.message,
    this.audioBase64,
    this.error,
  });

  factory TextChatResponse.fromJson(Map<String, dynamic> json) {
    return TextChatResponse(
      success: json['success'] ?? false,
      message: json['message'],
      audioBase64: json['audio_base64'],
      error: json['error'],
    );
  }
}

// Audio Chat Response Model
class AudioChatResponse {
  final bool success;
  final String? transcription;
  final String? message;
  final String? audioBase64;
  final String? error;

  AudioChatResponse({
    required this.success,
    this.transcription,
    this.message,
    this.audioBase64,
    this.error,
  });

  factory AudioChatResponse.fromJson(Map<String, dynamic> json) {
    return AudioChatResponse(
      success: json['success'] ?? false,
      transcription: json['transcription'],
      message: json['message'],
      audioBase64: json['audio_base64'],
      error: json['error'],
    );
  }
}

// User Profile Model
class UserProfile {
  final String name;
  final String email;
  final String id;
  final String currency;

  UserProfile({
    required this.name,
    required this.email,
    required this.id,
    required this.currency,
  });

  factory UserProfile.fromJson(Map<String, dynamic> json) {
    return UserProfile(
      name: json['name'] ?? '',
      email: json['email'] ?? '',
      id: json['id']?.toString() ?? '',
      currency: json['currency'] ?? 'USD',
    );
  }
}

// Balance Model
class Balance {
  final String person;
  final double amount;
  final String direction; // 'they_owe_you' or 'you_owe_them'
  final String amountFormatted;

  Balance({
    required this.person,
    required this.amount,
    required this.direction,
    required this.amountFormatted,
  });

  factory Balance.fromJson(Map<String, dynamic> json) {
    return Balance(
      person: json['person'] ?? '',
      amount: (json['amount'] ?? 0).toDouble(),
      direction: json['direction'] ?? '',
      amountFormatted: json['amount_formatted'] ?? '',
    );
  }
}

// Expenses Data Model
class ExpensesData {
  final List<Balance> balances;
  final int totalPeople;

  ExpensesData({
    required this.balances,
    required this.totalPeople,
  });

  factory ExpensesData.fromJson(Map<String, dynamic> json) {
    return ExpensesData(
      balances: (json['balances'] as List?)
          ?.map((e) => Balance.fromJson(e))
          .toList() ?? [],
      totalPeople: json['total_people'] ?? 0,
    );
  }
}
```

## 4. Audio Helper Class

```dart
import 'package:record/record.dart';
import 'package:audioplayers/audioplayers.dart';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

class AudioHelper {
  static final AudioRecorder _recorder = AudioRecorder();
  static final AudioPlayer _player = AudioPlayer();

  // Request microphone permission
  static Future<bool> requestMicrophonePermission() async {
    final status = await Permission.microphone.request();
    return status == PermissionStatus.granted;
  }

  // Start recording
  static Future<void> startRecording(String filePath) async {
    if (await _recorder.hasPermission()) {
      await _recorder.start(
        const RecordConfig(
          encoder: AudioEncoder.wav,
          sampleRate: 16000,
          bitRate: 128000,
        ),
        path: filePath,
      );
    }
  }

  // Stop recording
  static Future<String?> stopRecording() async {
    return await _recorder.stop();
  }

  // Play audio from base64
  static Future<void> playAudioFromBase64(String base64Audio) async {
    try {
      final bytes = base64Decode(base64Audio);
      final tempDir = await getTemporaryDirectory();
      final tempFile = File('${tempDir.path}/temp_audio.wav');
      await tempFile.writeAsBytes(bytes);
      
      await _player.play(DeviceFileSource(tempFile.path));
    } catch (e) {
      print('Error playing audio: $e');
    }
  }

  // Get temporary file path for recording
  static Future<String> getRecordingPath() async {
    final directory = await getTemporaryDirectory();
    return '${directory.path}/recording_${DateTime.now().millisecondsSinceEpoch}.wav';
  }
}
```

## 5. Usage Example Widget

```dart
import 'package:flutter/material.dart';

class VoiceChatScreen extends StatefulWidget {
  @override
  _VoiceChatScreenState createState() => _VoiceChatScreenState();
}

class _VoiceChatScreenState extends State<VoiceChatScreen> {
  final VoiceAssistantService _service = VoiceAssistantService();
  final TextEditingController _textController = TextEditingController();
  final List<ChatMessage> _messages = [];
  bool _isRecording = false;
  bool _isLoading = false;
  String? _recordingPath;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Voice Assistant')),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                final message = _messages[index];
                return ChatBubble(message: message);
              },
            ),
          ),
          _buildInputArea(),
        ],
      ),
    );
  }

  Widget _buildInputArea() {
    return Container(
      padding: EdgeInsets.all(16),
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _textController,
              decoration: InputDecoration(
                hintText: 'Type your message...',
                border: OutlineInputBorder(),
              ),
              onSubmitted: _sendTextMessage,
            ),
          ),
          SizedBox(width: 8),
          IconButton(
            onPressed: _sendTextMessage,
            icon: Icon(Icons.send),
          ),
          IconButton(
            onPressed: _isRecording ? _stopRecording : _startRecording,
            icon: Icon(_isRecording ? Icons.stop : Icons.mic),
            color: _isRecording ? Colors.red : Colors.blue,
          ),
        ],
      ),
    );
  }

  // Send text message
  void _sendTextMessage([String? text]) async {
    final message = text ?? _textController.text.trim();
    if (message.isEmpty) return;

    _textController.clear();
    _addMessage(ChatMessage(text: message, isUser: true));

    setState(() => _isLoading = true);

    try {
      final response = await _service.sendTextMessage(message);
      
      if (response.success && response.message != null) {
        _addMessage(ChatMessage(text: response.message!, isUser: false));
        
        // Play audio response if available
        if (response.audioBase64 != null) {
          AudioHelper.playAudioFromBase64(response.audioBase64!);
        }
      } else {
        _addMessage(ChatMessage(text: response.error ?? 'Error occurred', isUser: false));
      }
    } catch (e) {
      _addMessage(ChatMessage(text: 'Error: $e', isUser: false));
    }

    setState(() => _isLoading = false);
  }

  // Start voice recording
  void _startRecording() async {
    if (!await AudioHelper.requestMicrophonePermission()) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Microphone permission required')),
      );
      return;
    }

    _recordingPath = await AudioHelper.getRecordingPath();
    await AudioHelper.startRecording(_recordingPath!);
    
    setState(() => _isRecording = true);
  }

  // Stop voice recording and send
  void _stopRecording() async {
    final path = await AudioHelper.stopRecording();
    setState(() => _isRecording = false);

    if (path != null) {
      _addMessage(ChatMessage(text: '[Voice Message]', isUser: true));
      setState(() => _isLoading = true);

      try {
        final audioFile = File(path);
        final response = await _service.sendAudioMessage(audioFile);
        
        if (response.success) {
          if (response.transcription != null) {
            _addMessage(ChatMessage(text: 'You said: "${response.transcription}"', isUser: true, isTranscription: true));
          }
          
          if (response.message != null) {
            _addMessage(ChatMessage(text: response.message!, isUser: false));
            
            // Play audio response
            if (response.audioBase64 != null) {
              AudioHelper.playAudioFromBase64(response.audioBase64!);
            }
          }
        } else {
          _addMessage(ChatMessage(text: response.error ?? 'Error processing audio', isUser: false));
        }
      } catch (e) {
        _addMessage(ChatMessage(text: 'Error: $e', isUser: false));
      }

      setState(() => _isLoading = false);
    }
  }

  void _addMessage(ChatMessage message) {
    setState(() => _messages.add(message));
  }
}

// Chat Message Model
class ChatMessage {
  final String text;
  final bool isUser;
  final bool isTranscription;

  ChatMessage({
    required this.text,
    required this.isUser,
    this.isTranscription = false,
  });
}

// Chat Bubble Widget
class ChatBubble extends StatelessWidget {
  final ChatMessage message;

  const ChatBubble({required this.message});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: EdgeInsets.symmetric(vertical: 4, horizontal: 8),
      child: Row(
        mainAxisAlignment: message.isUser ? MainAxisAlignment.end : MainAxisAlignment.start,
        children: [
          Container(
            constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.7),
            padding: EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: message.isUser ? Colors.blue : Colors.grey[300],
              borderRadius: BorderRadius.circular(16),
            ),
            child: Text(
              message.text,
              style: TextStyle(
                color: message.isUser ? Colors.white : Colors.black,
                fontStyle: message.isTranscription ? FontStyle.italic : FontStyle.normal,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
```

## 6. API Endpoints Summary

| Endpoint | Method | Purpose | Request | Response |
|----------|--------|---------|---------|----------|
| `/api/chat/text` | POST | Send text message | `{message, language_code}` | `{success, message, audio_base64}` |
| `/api/chat/audio` | POST | Send audio file | `FormData(audio_file, language_code)` | `{success, transcription, message, audio_base64}` |
| `/api/user/profile` | GET | Get user profile | None | `{success, user: {name, email, id, currency}}` |
| `/api/expenses` | GET | Get expenses/balances | None | `{success, data: {balances, total_people}}` |
| `/api/health` | GET | Health check | None | `{status, message}` |

## 7. Integration Steps

1. **Add dependencies** to `pubspec.yaml`
2. **Copy the service class** and models to your Flutter project
3. **Replace `baseUrl`** with your server IP address
4. **Add permissions** to `android/app/src/main/AndroidManifest.xml`:
   ```xml
   <uses-permission android:name="android.permission.RECORD_AUDIO" />
   <uses-permission android:name="android.permission.INTERNET" />
   ```
5. **Use the VoiceChatScreen** widget in your app
6. **Test the integration** with both text and voice messages

The API provides the same functionality as your web interface but optimized for mobile Flutter apps with proper error handling and audio support.