import asyncio
import base64
import io
import json
import logging
import os
import tempfile
import time
import uuid
from typing import Dict, List, Optional, Tuple

from fastapi import (
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    FastAPI
)
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Import speech modules
from speech_modules import (
    process_audio_file,
    extract_audio_features,
    text_to_speech,
    speech_to_speech,
    process_file,
    check_speech_services,
    cleanup_temp_files,
    TTS_AVAILABLE,
    WHISPER_MODEL
)

# Initialize logger
logger = logging.getLogger(__name__)

# Create FastAPI app instance
app = FastAPI(
    title="Speech Processing API",
    description="API for speech-to-text, text-to-speech, and speech-to-speech processing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple models for basic functionality (without textgen.py dependencies)
class SimpleModelConfig:
    def __init__(self, temperature=0.7):
        self.temperature = temperature

# Global variable for intent statistics
intent_statistics = {"conversation": 0, "question": 0, "command": 0}

@app.post("/text-to-speech")
async def generate_speech(
    text: str = Form(..., description="Text to convert to speech"),
    output_format: str = Form("mp3", description="Audio format (mp3 supported)"),
    language: str = Form("en", description="Language code"),
    voice: str = Form("default", description="Voice to use (if available)"),
):
    """Convert text to speech and return audio file"""
    try:
        # Check if TTS is available
        if not TTS_AVAILABLE:
            return {
                "status": "error",
                "message": "Text-to-speech not available. Check server logs for details.",
                "setup_help": "Make sure gTTS is installed with 'pip install gtts'",
            }

        # Generate speech
        result = text_to_speech(text, "mp3", language, voice)

        # Create response
        content_disposition = f"attachment; filename=speech.{result['format']}"

        # Return audio file
        return Response(
            content=result["audio_data"],
            media_type=f"audio/{result['format']}",
            headers={"Content-Disposition": content_disposition},
        )
    except Exception as e:
        logger.error(f"Error in generate_speech: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {str(e)}")


@app.post("/transcribe-audio")
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file to transcribe"),
    model_size: str = Form(
        "base", description="Whisper model size (tiny, base, small, medium, large)"
    ),
    language: Optional[str] = Form(
        None, description="Language code (optional, auto-detect if None)"
    ),
    analyze: bool = Form(False, description="Analyze audio content after transcription"),
):
    """Transcribe audio file to text using OpenAI's Whisper model"""
    try:
        # Check if Whisper is available
        if WHISPER_MODEL is None:
            return {
                "status": "error",
                "message": "Whisper model not available. Check server logs for details.",
                "setup_help": "Make sure OpenAI Whisper is installed with 'pip install openai-whisper'",
            }

        # Check if file is an audio file
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in [".mp3", ".wav", ".ogg", ".flac", ".m4a"]:
            raise HTTPException(
                status_code=400, detail="Unsupported file format. Please upload an audio file."
            )

        # Read file content
        file_content = await file.read()

        # Process the audio file
        processed = process_file(UploadFile(file=io.BytesIO(file_content), filename=file.filename))

        if processed.get("type") == "error":
            raise HTTPException(status_code=500, detail=processed.get("error"))

        transcription = processed.get("transcription", "")

        # Check if transcription failed
        if not transcription or transcription.startswith("[Error:"):
            return {
                "status": "error",
                "message": (
                    transcription if transcription else "Transcription failed with unknown error"
                ),
                "filename": file.filename,
            }

        # Prepare response
        response = {
            "status": "success",
            "filename": file.filename,
            "transcription": transcription,
            "audio_details": {
                "duration_seconds": processed.get("duration", 0),
                "sample_rate": processed.get("sample_rate", 0),
            },
        }

        # Simple analysis if requested (without LLM dependency)
        if analyze and transcription and len(transcription) > 0:
            try:
                # Basic text analysis
                word_count = len(transcription.split())
                char_count = len(transcription)
                
                response["analysis"] = {
                    "word_count": word_count,
                    "character_count": char_count,
                    "estimated_speaking_time": f"{word_count / 150:.1f} minutes",  # Average speaking rate
                    "summary": f"Audio contains {word_count} words ({char_count} characters) of transcribed speech."
                }

            except Exception as analysis_error:
                logger.error(f"Error analyzing transcription: {str(analysis_error)}")
                response["analysis_error"] = str(analysis_error)

        return response

    except Exception as e:
        logger.error(f"Error in transcribe_audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/audio-to-response")
async def audio_to_response(
    file: UploadFile = File(..., description="Audio file containing the user's query"),
    language: str = Form("en", description="Language code for response"),
    generate_audio_response: bool = Form(
        False, description="Whether to generate audio for the response"
    ),
    temperature: float = Form(0.7, description="Temperature for generation"),
    show_progress: bool = Form(True, description="Show generation progress"),
):
    """Process audio file, transcribe it, and generate a simple response"""
    try:
        # Check if Whisper is available for transcription
        if WHISPER_MODEL is None:
            return {
                "status": "error",
                "message": "Whisper model not available for transcription. Check server logs for details.",
                "setup_help": "Make sure OpenAI Whisper is installed with 'pip install openai-whisper'",
            }

        # Check if the file is an audio file
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in [".mp3", ".wav", ".ogg", ".flac", ".m4a"]:
            raise HTTPException(
                status_code=400, detail="Unsupported file format. Please upload an audio file."
            )

        # Read file content
        file_content = await file.read()

        # Step 1: Transcribe the audio
        logger.info(f"Transcribing audio file: {file.filename}")
        processed = process_file(UploadFile(file=io.BytesIO(file_content), filename=file.filename))

        if processed.get("type") == "error":
            raise HTTPException(status_code=500, detail=processed.get("error"))

        transcription = processed.get("transcription", "")

        # Check if transcription failed
        if not transcription or transcription.startswith("[Error:"):
            return {
                "status": "error",
                "message": (
                    transcription if transcription else "Transcription failed with unknown error"
                ),
                "filename": file.filename,
            }

        logger.info(f"Transcription successful: {len(transcription)} characters")

        # Step 2: Generate simple response (without LLM)
        # For now, we'll just echo back the transcription with some basic processing
        simple_response = f"I heard you say: '{transcription}'. This is a simple echo response since no LLM is configured."

        # Update intent statistics
        intent_statistics["conversation"] += 1

        # Prepare response object
        response_data = {
            "status": "success",
            "transcribed_audio": transcription,
            "response": simple_response,
            "model_used": "simple_echo",
            "language": language,
            "intent": "conversation",
            "audio_details": {
                "duration_seconds": processed.get("duration", 0),
                "sample_rate": processed.get("sample_rate", 0),
                "filename": file.filename,
            },
            "note": "This is a simple echo response. For full LLM responses, integrate with textgen.py",
        }

        # Step 3: Generate audio response if requested
        if generate_audio_response and TTS_AVAILABLE:
            try:
                logger.info("Generating audio response")
                audio_result = text_to_speech(simple_response, "mp3", language)

                # Convert audio to base64 for inclusion in response
                audio_b64 = base64.b64encode(audio_result["audio_data"]).decode("utf-8")

                # Add audio to response
                response_data["audio_response"] = {
                    "format": audio_result["format"],
                    "audio_base64": audio_b64,
                    "player_html": f'<audio controls src="data:audio/{audio_result["format"]};base64,{audio_b64}"></audio>',
                }
            except Exception as tts_error:
                logger.error(f"Error generating audio response: {str(tts_error)}")
                response_data["tts_error"] = str(tts_error)

        return response_data

    except Exception as e:
        logger.error(f"Error in audio_to_response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/audio-response-only")
async def audio_response_only(
    file: UploadFile = File(..., description="Audio file containing the user's query"),
    language: str = Form("en", description="Language code for response"),
    temperature: float = Form(0.7, description="Temperature for generation"),
    show_progress: bool = Form(True, description="Show generation progress"),
):
    """Process audio and return an audio response directly (for voice assistants)"""
    try:
        # Check both required services
        if WHISPER_MODEL is None:
            raise HTTPException(status_code=503, detail="Speech-to-text service not available")

        if not TTS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Text-to-speech service not available")

        # Read file content
        file_content = await file.read()

        # Transcribe the audio
        print(f"\nTranscribing audio file: {file.filename}...")
        transcribe_start = time.time()
        processed = process_file(UploadFile(file=io.BytesIO(file_content), filename=file.filename))
        transcribe_time = time.time() - transcribe_start

        if processed.get("type") == "error" or processed.get("type") != "audio":
            raise HTTPException(status_code=400, detail="Failed to process audio file")

        transcription = processed.get("transcription", "")
        print(f"Transcription complete ({transcribe_time:.2f}s, {len(transcription)} chars) ✓")

        # Generate simple text response (without LLM)
        simple_response = f"I heard you say: '{transcription}'. This is a simple echo response."

        # Convert to audio
        audio_result = text_to_speech(simple_response, "mp3", language)

        # Return audio directly as a downloadable file
        return Response(
            content=audio_result["audio_data"],
            media_type=f"audio/{audio_result['format']}",
            headers={
                "Content-Disposition": f"attachment; filename=response.{audio_result['format']}"
            },
        )

    except Exception as e:
        logger.error(f"Error in audio_response_only: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speech-to-speech")
async def speech_to_speech_endpoint(
    file: UploadFile = File(..., description="Audio file to process"),
    language: str = Form("en", description="Language code for processing"),
    output_format: str = Form("mp3", description="Output audio format"),
):
    """Speech-to-speech endpoint: process audio input and return audio response"""
    try:
        # Check if both services are available
        if WHISPER_MODEL is None:
            raise HTTPException(status_code=503, detail="Speech-to-text service not available")

        if not TTS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Text-to-speech service not available")

        # Check if file is an audio file
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in [".mp3", ".wav", ".ogg", ".flac", ".m4a"]:
            raise HTTPException(
                status_code=400, detail="Unsupported file format. Please upload an audio file."
            )

        # Read file content
        file_content = await file.read()

        # Process speech-to-speech
        result = speech_to_speech(file_content, file_ext, language, output_format)

        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])

        # Return audio response
        return Response(
            content=result["audio_data"],
            media_type=f"audio/{result['format']}",
            headers={
                "Content-Disposition": f"attachment; filename=speech_to_speech.{result['format']}"
            },
        )

    except Exception as e:
        logger.error(f"Error in speech_to_speech_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tts-setup-help")
async def tts_setup_help():
    """Get help with setting up text-to-speech"""
    return {
        "tts_status": "Available" if TTS_AVAILABLE else "Not available",
        "installation_instructions": [
            "1. Install gTTS: pip install gtts",
            "2. Install additional Python packages: pip install soundfile",
            "3. For better audio quality, consider installing: pip install pydub",
        ],
        "troubleshooting": [
            "If TTS is not working, check that gTTS is properly installed",
            "Ensure you have an internet connection for gTTS to work",
            "Check server logs for detailed error messages",
        ],
    }


@app.get("/whisper-setup-help")
async def whisper_setup_help():
    """Get help with setting up Whisper for speech recognition"""
    return {
        "whisper_status": "Available" if WHISPER_MODEL is not None else "Not available",
        "installation_instructions": [
            "1. Install Whisper: pip install openai-whisper",
            "2. Install additional dependencies: pip install pydub soundfile",
            "3. For better performance, install ffmpeg:",
            "   - Windows: Download from https://ffmpeg.org/download.html",
            "   - macOS: brew install ffmpeg",
            "   - Ubuntu/Debian: sudo apt install ffmpeg",
        ],
        "troubleshooting": [
            "If Whisper is not working, check that it's properly installed",
            "Ensure ffmpeg is available in your system PATH",
            "Check server logs for detailed error messages",
        ],
    }


@app.get("/speech-services-status")
async def speech_services_status():
    """Check the status of all speech processing services"""
    return check_speech_services()


@app.get("/intent-stats")
async def get_intent_statistics():
    """Get statistics on detected intents"""
    total = sum(intent_statistics.values()) or 1  # Avoid division by zero

    intent_data = []
    for intent, count in intent_statistics.items():
        intent_data.append(
            {
                "intent": intent,
                "count": count,
                "percentage": round((count / total) * 100, 2),
            }
        )

    # Sort by count, descending
    intent_data.sort(key=lambda x: x["count"], reverse=True)

    return {"total_requests": total, "intents": intent_data}


# Cleanup endpoint for temporary files
@app.post("/cleanup-temp-files")
async def cleanup_temp_files_endpoint():
    """Clean up old temporary files"""
    try:
        cleanup_temp_files()
        return {"status": "success", "message": "Temporary files cleaned up successfully"}
    except Exception as e:
        logger.error(f"Error in cleanup_temp_files_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Speech Processing API",
        "version": "1.0.0",
        "description": "Standalone API for speech-to-text, text-to-speech, and speech-to-speech processing",
        "endpoints": [
            "/text-to-speech - Convert text to speech",
            "/transcribe-audio - Convert speech to text",
            "/audio-to-response - Process audio and generate response",
            "/audio-response-only - Process audio and return audio response",
            "/speech-to-speech - Process audio input and return audio output",
            "/speech-services-status - Check service availability",
            "/tts-setup-help - Get TTS setup help",
            "/whisper-setup-help - Get Whisper setup help",
            "/intent-stats - Get usage statistics",
            "/cleanup-temp-files - Clean up temporary files"
        ],
        "note": "This is a standalone speech API. For full LLM functionality, integrate with textgen.py"
    }


# Export the endpoints for use in other modules
__all__ = [
    "generate_speech",
    "transcribe_audio", 
    "audio_to_response",
    "audio_response_only",
    "speech_to_speech_endpoint",
    "tts_setup_help",
    "whisper_setup_help",
    "speech_services_status",
    "cleanup_temp_files_endpoint",
    "get_intent_statistics",
    "root"
]
