# Speech Processing API - Standalone Version

This is a standalone speech processing API that provides speech-to-text, text-to-speech, and speech-to-speech functionality without requiring the main `textgen.py` file.

# Speech Modules

This repository contains various modules for speech processing, including speech recognition, audio preprocessing, and related utilities.

## Features

- Speech-to-text conversion
- Audio preprocessing and feature extraction
- Utility functions for audio file handling
- ## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r speech_requirements.txt
```

### 2. Run the API Server

```bash
python run_speech_api.py
```

The server will start on `http://localhost:8001` by default.

### 3. Access the API

- **API Documentation**: http://localhost:8001/docs
- **Interactive API**: http://localhost:8001/redoc
- **API Root**: http://localhost:8001/

## 📋 Available Endpoints

### Core Speech Processing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/text-to-speech` | POST | Convert text to speech audio |
| `/transcribe-audio` | POST | Convert speech audio to text |
| `/audio-to-response` | POST | Process audio and generate response |
| `/audio-response-only` | POST | Process audio and return audio response |
| `/speech-to-speech` | POST | Process audio input and return audio output |

### Utility Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/speech-services-status` | GET | Check availability of speech services |
| `/tts-setup-help` | GET | Get text-to-speech setup help |
| `/whisper-setup-help` | GET | Get Whisper setup help |
| `/intent-stats` | GET | Get usage statistics |
| `/cleanup-temp-files` | POST | Clean up temporary files |

## 🔧 Configuration

### Environment Variables

```bash
# Server configuration
export SPEECH_API_HOST="0.0.0.0"  # Default: 0.0.0.0
export SPEECH_API_PORT="8001"      # Default: 8001

# FFmpeg path (if needed)
export FFMPEG_PATH="/usr/bin/ffmpeg"  # Default: ffmpeg
```

### Port Configuration

The speech API runs on port **8001** by default to avoid conflicts with other services. You can change this by setting the `SPEECH_API_PORT` environment variable.

## 📁 File Structure

```
├── speech_modules.py          # Core speech processing functions
├── speech_endpoints.py        # FastAPI endpoints and server
├── run_speech_api.py          # Server startup script
├── speech_requirements.txt    # Python dependencies
├── SPEECH_README.md          # This file
└── temp_uploads/             # Temporary file storage (auto-created)
```

## 🎯 Usage Examples

### 1. Text to Speech

```bash
curl -X POST "http://localhost:8001/text-to-speech" \
  -F "text=Hello, this is a test message" \
  -F "language=en" \
  -F "output_format=mp3" \
  --output speech.mp3
```

### 2. Speech to Text (Transcription)

```bash
curl -X POST "http://localhost:8001/transcribe-audio" \
  -F "file=@audio_file.mp3" \
  -F "language=en" \
  -F "analyze=true"
```

### 3. Audio to Response

```bash
curl -X POST "http://localhost:8001/audio-to-response" \
  -F "file=@audio_file.mp3" \
  -F "language=en" \
  -F "generate_audio_response=true"
```

### 4. Speech to Speech

```bash
curl -X POST "http://localhost:8001/speech-to-speech" \
  -F "file=@audio_file.mp3" \
  -F "language=en" \
  -F "output_format=mp3" \
  --output response.mp3
```

## 🔍 Testing the API

### 1. Check Service Status

```bash
curl http://localhost:8001/speech-services-status
```

### 2. View API Documentation

Open your browser and go to:
- http://localhost:8001/docs (Swagger UI)
- http://localhost:8001/redoc (ReDoc)

### 3. Test with Python

```python
import requests

# Test text-to-speech
response = requests.post(
    "http://localhost:8001/text-to-speech",
    data={
        "text": "Hello, world!",
        "language": "en",
        "output_format": "mp3"
    }
)

if response.status_code == 200:
    with open("test_speech.mp3", "wb") as f:
        f.write(response.content)
    print("Speech generated successfully!")
```

## ⚠️ Important Notes

### 1. Dependencies

- **gTTS**: Requires internet connection for text-to-speech
- **Whisper**: Requires sufficient disk space for model downloads
- **FFmpeg**: Required for audio format conversion

### 2. Limitations

- **No LLM Integration**: This standalone version provides simple echo responses instead of intelligent LLM responses
- **Basic Analysis**: Audio analysis is limited to basic statistics (word count, duration, etc.)
- **No Conversation Memory**: Each request is processed independently

### 3. For Full Functionality

To get full LLM-powered responses and conversation memory, integrate with the main `textgen.py` file by:

1. Importing the speech modules into `textgen.py`
2. Using the speech functions within the LLM workflow
3. Running the combined application

## 🐛 Troubleshooting

### Common Issues

1. **TTS Not Working**
   - Check internet connection
   - Verify gTTS installation: `pip install gtts`
   - Check server logs for error messages

2. **Whisper Not Working**
   - Verify Whisper installation: `pip install openai-whisper`
   - Check FFmpeg installation
   - Ensure sufficient disk space for models

3. **Port Already in Use**
   - Change port: `export SPEECH_API_PORT="8002"`
   - Or kill existing process using the port

4. **Audio Format Issues**
   - Supported formats: MP3, WAV, OGG, FLAC, M4A
   - Install additional codecs if needed

### Getting Help

- Check the `/tts-setup-help` and `/whisper-setup-help` endpoints
- Review server logs for detailed error messages
- Verify all dependencies are properly installed

## 🔄 Integration with Main Application

When you're ready to integrate with the full LLM application:

1. **Copy speech modules**: The speech functions can be imported into `textgen.py`
2. **Update endpoints**: Modify the main FastAPI app to include speech endpoints
3. **Enhanced responses**: Use LLM models for intelligent audio responses
4. **Conversation memory**: Enable full conversation tracking and context

## 📝 License

This speech processing API is part of the larger LLM application and follows the same licensing terms.
