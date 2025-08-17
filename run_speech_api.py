#!/usr/bin/env python3
"""
Run script for the standalone Speech Processing API
This script starts the FastAPI server for speech-to-text, text-to-speech, and speech-to-speech functionality
"""

import uvicorn
import os
import sys

def main():
    """Main function to run the speech API server"""
    
    # Set default configuration
    host = os.environ.get("SPEECH_API_HOST", "0.0.0.0")
    port = int(os.environ.get("SPEECH_API_PORT", "8001"))  # Different port to avoid conflicts
    
    print("🚀 Starting Speech Processing API Server...")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🌐 API URL: http://{host}:{port}")
    print(f"📚 Documentation: http://{host}:{port}/docs")
    print(f"🔍 Interactive API: http://{host}:{port}/redoc")
    print()
    print("Available endpoints:")
    print("  POST /text-to-speech     - Convert text to speech")
    print("  POST /transcribe-audio   - Convert speech to text")
    print("  POST /audio-to-response  - Process audio and generate response")
    print("  POST /audio-response-only - Process audio and return audio response")
    print("  POST /speech-to-speech   - Process audio input and return audio output")
    print("  GET  /speech-services-status - Check service availability")
    print("  GET  /tts-setup-help     - Get TTS setup help")
    print("  GET  /whisper-setup-help - Get Whisper setup help")
    print("  GET  /intent-stats       - Get usage statistics")
    print("  GET  /cleanup-temp-files - Clean up temporary files")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Start the server
        uvicorn.run(
            "speech_endpoints:app",
            host=host,
            port=port,
            reload=True,  # Enable auto-reload for development
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
