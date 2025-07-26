class VoiceAssistantChat {
    constructor() {
        this.ws = null;
        this.isVoiceMode = false;
        this.isListening = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.stream = null;
        this.audioContext = null;
        this.analyser = null;
        this.isProcessing = false;
        this.currentAudio = null;
        
        // Advanced VAD (Voice Activity Detection) settings
        this.vadSettings = {
            silenceThreshold: 0.01,     // Volume threshold for silence
            silenceDuration: 1500,      // ms of silence before processing
            minSpeechDuration: 500,     // Minimum speech duration to process
            maxSpeechDuration: 30000,   // Maximum speech duration
            volumeSmoothing: 0.2        // Smoothing factor for volume
        };
        
        // VAD state
        this.vadState = {
            isSpeaking: false,
            speechStartTime: 0,
            lastSpeechTime: 0,
            silenceStartTime: 0,
            smoothedVolume: 0,
            volumeHistory: []
        };
        
        // Audio processing
        this.processor = null;
        this.isRecordingActive = false;
        this.recordingStartTime = 0;
        
        this.initializeElements();
        this.setupEventListeners();
        this.connect();
    }
    
    initializeElements() {
        this.statusEl = document.getElementById('status');
        this.messagesEl = document.getElementById('messages');
        this.messageInput = document.getElementById('messageInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.typingEl = document.getElementById('typing');
        
        // Mode controls
        this.textModeBtn = document.getElementById('textModeBtn');
        this.voiceModeBtn = document.getElementById('voiceModeBtn');
        this.textMode = document.getElementById('textMode');
        this.voiceControls = document.getElementById('voiceControls');
        this.startVoiceBtn = document.getElementById('startVoiceBtn');
        this.stopVoiceBtn = document.getElementById('stopVoiceBtn');
        this.volumeBar = document.getElementById('volumeBar');
    }
    
    setupEventListeners() {
        // Text mode
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
        
        // Mode switching
        this.textModeBtn.addEventListener('click', () => this.switchToTextMode());
        this.voiceModeBtn.addEventListener('click', () => this.switchToVoiceMode());
        
        // Voice controls
        this.startVoiceBtn.addEventListener('click', () => this.startVoiceChat());
        this.stopVoiceBtn.addEventListener('click', () => this.stopVoiceChat());
    }
    
    switchToTextMode() {
        this.isVoiceMode = false;
        this.textModeBtn.classList.add('active');
        this.voiceModeBtn.classList.remove('active');
        this.textMode.style.display = 'block';
        this.voiceControls.style.display = 'none';
        this.stopVoiceChat();
        this.updateStatus('Text Mode - Type your messages', 'connected');
    }
    
    switchToVoiceMode() {
        this.isVoiceMode = true;
        this.voiceModeBtn.classList.add('active');
        this.textModeBtn.classList.remove('active');
        this.textMode.style.display = 'none';
        this.voiceControls.style.display = 'block';
        this.updateStatus('Voice Mode - Click to start conversation', 'voice-active');
    }
    
    async startVoiceChat() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    sampleRate: 16000,
                    channelCount: 1
                }
            });
            
            await this.setupAdvancedAudioProcessing();
            this.startContinuousListening();
            
            this.startVoiceBtn.style.display = 'none';
            this.stopVoiceBtn.style.display = 'inline-block';
            this.updateStatus('Voice chat active - Speak naturally!', 'listening');
            this.addMessage('system', 'Advanced voice chat started! I\'ll respond instantly when you pause. You can interrupt me anytime.');
            
        } catch (error) {
            console.error('Error starting voice chat:', error);
            this.addMessage('system', 'Error: Could not access microphone');
        }
    }
    
    stopVoiceChat() {
        this.isListening = false;
        this.isRecordingActive = false;
        
        if (this.processor) {
            this.processor.disconnect();
            this.processor = null;
        }
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
        }
        
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        if (this.currentAudio) {
            this.currentAudio.pause();
            this.currentAudio = null;
        }
        
        // Reset VAD state
        this.vadState = {
            isSpeaking: false,
            speechStartTime: 0,
            lastSpeechTime: 0,
            silenceStartTime: 0,
            smoothedVolume: 0,
            volumeHistory: []
        };
        
        this.isProcessing = false;
        this.startVoiceBtn.style.display = 'inline-block';
        this.stopVoiceBtn.style.display = 'none';
        this.volumeBar.style.width = '0%';
        
        if (this.isVoiceMode) {
            this.updateStatus('Voice Mode - Click to start conversation', 'voice-active');
        }
    }
    
    async setupAdvancedAudioProcessing() {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000
        });
        
        // Resume context if suspended
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
        
        // Create audio processing chain
        const source = this.audioContext.createMediaStreamSource(this.stream);
        this.analyser = this.audioContext.createAnalyser();
        
        // Configure analyser for better VAD
        this.analyser.fftSize = 2048;
        this.analyser.smoothingTimeConstant = 0.3;
        this.analyser.minDecibels = -90;
        this.analyser.maxDecibels = -10;
        
        // Create script processor for real-time audio analysis
        this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
        
        // Connect the audio processing chain
        source.connect(this.analyser);
        source.connect(this.processor);
        this.processor.connect(this.audioContext.destination);
        
        // Set up real-time audio processing
        this.processor.onaudioprocess = (event) => {
            if (!this.isListening) return;
            this.processAudioFrame(event.inputBuffer);
        };
        
        // Start visual feedback
        this.startVolumeVisualization();
    }
    
    startContinuousListening() {
        this.isListening = true;
        this.isRecordingActive = false;
        this.audioChunks = [];
        
        // Initialize MediaRecorder for when we detect speech
        this.mediaRecorder = new MediaRecorder(this.stream, {
            mimeType: 'audio/webm;codecs=opus'
        });
        
        this.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                this.audioChunks.push(event.data);
            }
        };
        
        this.mediaRecorder.onstop = () => {
            if (this.audioChunks.length > 0 && !this.isProcessing) {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                this.processAudio(audioBlob);
            }
        };
        
        console.log('Continuous listening started with advanced VAD');
    }
    
    processAudioFrame(inputBuffer) {
        const inputData = inputBuffer.getChannelData(0);
        const currentTime = Date.now();
        
        // Calculate RMS (Root Mean Square) for volume detection
        let sum = 0;
        for (let i = 0; i < inputData.length; i++) {
            sum += inputData[i] * inputData[i];
        }
        const rms = Math.sqrt(sum / inputData.length);
        
        // Apply smoothing to reduce noise
        this.vadState.smoothedVolume = 
            this.vadSettings.volumeSmoothing * rms + 
            (1 - this.vadSettings.volumeSmoothing) * this.vadState.smoothedVolume;
        
        // Voice Activity Detection
        const isSpeechDetected = this.vadState.smoothedVolume > this.vadSettings.silenceThreshold;
        
        if (isSpeechDetected) {
            this.handleSpeechDetected(currentTime);
        } else {
            this.handleSilenceDetected(currentTime);
        }
        
        // Update volume history for better detection
        this.vadState.volumeHistory.push(this.vadState.smoothedVolume);
        if (this.vadState.volumeHistory.length > 10) {
            this.vadState.volumeHistory.shift();
        }
    }
    
    handleSpeechDetected(currentTime) {
        // If we detect speech while assistant is speaking, interrupt it
        if (this.currentAudio && !this.currentAudio.paused) {
            console.log('User interruption detected - stopping assistant');
            this.currentAudio.pause();
            this.currentAudio = null;
            this.isProcessing = false;
            this.updateStatus('You interrupted - I\'m listening...', 'listening');
        }
        
        if (!this.vadState.isSpeaking) {
            // Speech just started
            this.vadState.isSpeaking = true;
            this.vadState.speechStartTime = currentTime;
            this.vadState.silenceStartTime = 0;
            
            if (!this.isRecordingActive && !this.isProcessing) {
                this.startRecording();
            }
            
            this.updateStatus('Listening to you...', 'listening');
        }
        
        this.vadState.lastSpeechTime = currentTime;
    }
    
    handleSilenceDetected(currentTime) {
        if (this.vadState.isSpeaking) {
            if (this.vadState.silenceStartTime === 0) {
                this.vadState.silenceStartTime = currentTime;
            }
            
            const silenceDuration = currentTime - this.vadState.silenceStartTime;
            const speechDuration = this.vadState.lastSpeechTime - this.vadState.speechStartTime;
            
            // Check if we should process the speech
            if (silenceDuration >= this.vadSettings.silenceDuration && 
                speechDuration >= this.vadSettings.minSpeechDuration) {
                
                console.log(`Speech detected: ${speechDuration}ms, Silence: ${silenceDuration}ms - Processing...`);
                this.vadState.isSpeaking = false;
                this.stopRecording();
            }
        }
    }
    
    startRecording() {
        if (this.isRecordingActive || this.isProcessing) return;
        
        this.audioChunks = [];
        this.isRecordingActive = true;
        this.recordingStartTime = Date.now();
        
        try {
            this.mediaRecorder.start(100);
            console.log('Recording started');
        } catch (error) {
            console.error('Error starting recording:', error);
            this.isRecordingActive = false;
        }
    }
    
    stopRecording() {
        if (!this.isRecordingActive) return;
        
        this.isRecordingActive = false;
        
        try {
            if (this.mediaRecorder.state === 'recording') {
                this.mediaRecorder.stop();
                console.log('Recording stopped');
            }
        } catch (error) {
            console.error('Error stopping recording:', error);
        }
    }
    
    startVolumeVisualization() {
        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        const updateVisuals = () => {
            if (!this.isListening) return;
            
            this.analyser.getByteFrequencyData(dataArray);
            const average = dataArray.reduce((a, b) => a + b) / bufferLength;
            const volume = (average / 255) * 100;
            
            // Enhanced volume visualization
            const displayVolume = Math.max(volume, this.vadState.smoothedVolume * 1000);
            this.volumeBar.style.width = Math.min(displayVolume, 100) + '%';
            
            // Color coding based on speech detection
            if (this.vadState.isSpeaking) {
                this.volumeBar.style.background = 'linear-gradient(90deg, #28a745, #20c997)';
            } else {
                this.volumeBar.style.background = 'linear-gradient(90deg, #6c757d, #adb5bd)';
            }
            
            requestAnimationFrame(updateVisuals);
        };
        
        updateVisuals();
    }
    
    async processAudio(audioBlob) {
        if (this.isProcessing) return;
        
        this.isProcessing = true;
        this.updateStatus('Processing your speech...', 'voice-active');
        this.showTyping();
        
        try {
            const reader = new FileReader();
            reader.onload = () => {
                const base64Audio = reader.result.split(',')[1];
                this.ws.send(JSON.stringify({
                    type: 'audio',
                    audio: base64Audio
                }));
            };
            reader.readAsDataURL(audioBlob);
        } catch (error) {
            console.error('Error processing audio:', error);
            this.addMessage('system', 'Error processing audio');
            this.isProcessing = false;
            if (this.isVoiceMode && this.isListening) {
                this.updateStatus('Voice chat active - Speak naturally!', 'listening');
            }
        }
    }
    
    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/chat-ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            this.updateStatus('Connected - Text Mode', 'connected');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
        
        this.ws.onclose = () => {
            this.updateStatus('Disconnected', 'disconnected');
            this.addMessage('system', 'Connection lost. Attempting to reconnect...');
            setTimeout(() => this.connect(), 3000);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateStatus('Connection Error', 'disconnected');
        };
    }
    
    handleMessage(data) {
        this.hideTyping();
        
        switch(data.type) {
            case 'response':
                this.addMessage('assistant', data.message);
                break;
            case 'transcription':
                this.addMessage('user', `${data.message}`);
                break;
            case 'audio_response':
                this.playAudioResponse(data.audio);
                break;
            case 'error':
                this.addMessage('system', `Error: ${data.message}`);
                this.isProcessing = false;
                if (this.isVoiceMode && this.isListening) {
                    this.updateStatus('Voice chat active - Speak naturally!', 'listening');
                }
                break;
        }
    }
    
    playAudioResponse(base64Audio) {
        try {
            const audioBlob = this.base64ToBlob(base64Audio, 'audio/wav');
            const audioUrl = URL.createObjectURL(audioBlob);
            
            // Stop any currently playing audio
            if (this.currentAudio) {
                this.currentAudio.pause();
                this.currentAudio = null;
            }
            
            this.currentAudio = new Audio(audioUrl);
            this.updateStatus('Assistant speaking - you can interrupt anytime', 'voice-active');
            
            // Enhanced audio event handling
            this.currentAudio.onloadstart = () => {
                console.log('Audio loading started');
            };
            
            this.currentAudio.oncanplay = () => {
                console.log('Audio ready to play');
            };
            
            this.currentAudio.onplay = () => {
                console.log('Audio playback started');
            };
            
            this.currentAudio.onended = () => {
                console.log('Audio playback ended naturally');
                URL.revokeObjectURL(audioUrl);
                this.currentAudio = null;
                this.isProcessing = false;
                
                if (this.isVoiceMode && this.isListening) {
                    this.updateStatus('Voice chat active - Speak naturally!', 'listening');
                    // Reset VAD state for next interaction
                    this.vadState.isSpeaking = false;
                    this.vadState.silenceStartTime = 0;
                }
            };
            
            this.currentAudio.onpause = () => {
                console.log('Audio playback paused (likely interrupted)');
                URL.revokeObjectURL(audioUrl);
                this.currentAudio = null;
                this.isProcessing = false;
                
                if (this.isVoiceMode && this.isListening) {
                    this.updateStatus('Voice chat active - Continue speaking!', 'listening');
                }
            };
            
            this.currentAudio.onerror = (error) => {
                console.error('Error playing audio response:', error);
                URL.revokeObjectURL(audioUrl);
                this.currentAudio = null;
                this.isProcessing = false;
                
                if (this.isVoiceMode && this.isListening) {
                    this.updateStatus('Voice chat active - Speak naturally!', 'listening');
                }
            };
            
            // Start playback
            this.currentAudio.play().catch(error => {
                console.error('Error starting audio playback:', error);
                this.isProcessing = false;
            });
            
        } catch (error) {
            console.error('Error setting up audio response:', error);
            this.isProcessing = false;
            if (this.isVoiceMode && this.isListening) {
                this.updateStatus('Voice chat active - Speak naturally!', 'listening');
            }
        }
    }
    
    base64ToBlob(base64, mimeType) {
        const byteCharacters = atob(base64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        return new Blob([byteArray], { type: mimeType });
    }
    
    sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        
        this.addMessage('user', message);
        this.showTyping();
        
        this.ws.send(JSON.stringify({
            type: 'text',
            message: message
        }));
        
        this.messageInput.value = '';
    }
    
    addMessage(type, content) {
        const messageEl = document.createElement('div');
        messageEl.className = `message ${type}`;
        messageEl.textContent = content;
        
        this.messagesEl.appendChild(messageEl);
        this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
    }
    
    showTyping() {
        this.typingEl.classList.add('show');
        this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
    }
    
    hideTyping() {
        this.typingEl.classList.remove('show');
    }
    
    updateStatus(message, className) {
        this.statusEl.textContent = message;
        this.statusEl.className = `status ${className}`;
    }
}

// Initialize the chat when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new VoiceAssistantChat();
});