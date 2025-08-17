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

import soundfile as sf
import whisper
from fastapi import (
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from pydub import AudioSegment

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize TTS model
try:
    from gtts import gTTS
    logger.info("gTTS module loaded successfully")
    TTS_AVAILABLE = True
except ImportError as import_error:
    logger.error(f"gTTS module not found: {str(import_error)}")
    TTS_AVAILABLE = False
except Exception as e:
    logger.error(f"Failed to initialize gTTS: {str(e)}")
    TTS_AVAILABLE = False

# Initialize Whisper model
try:
    import whisper
    WHISPER_MODEL = None
    
    # Try to load a small model by default for faster inference
    try:
        print("Loading Whisper model (this may take a few minutes on first run)...")
        WHISPER_MODEL = whisper.load_model("tiny")  # Start with tiny model for faster loading
        logger.info("Whisper model loaded successfully: tiny")
    except Exception as model_error:
        logger.warning(f"Could not load Whisper tiny model: {str(model_error)}")
        try:
            # Try base model if tiny fails
            print("Trying base model...")
            WHISPER_MODEL = whisper.load_model("base")
            logger.info("Whisper model loaded successfully: base")
        except Exception as base_error:
            logger.error(f"Failed to load Whisper base model: {str(base_error)}")
            try:
                # Last resort: try to download a fresh copy
                print("Attempting to download fresh Whisper model...")
                import shutil
                import os
                
                # Clear the cache directory to force fresh download
                cache_dir = os.path.expanduser("~/.cache/whisper")
                if os.path.exists(cache_dir):
                    print(f"Clearing Whisper cache at: {cache_dir}")
                    shutil.rmtree(cache_dir)
                
                # Try loading again
                WHISPER_MODEL = whisper.load_model("tiny")
                logger.info("Whisper model loaded successfully after cache clear: tiny")
            except Exception as final_error:
                logger.error(f"All Whisper model loading attempts failed: {str(final_error)}")
                WHISPER_MODEL = None
                print("❌ Whisper model could not be loaded. Speech-to-text will not work.")
                print("   Please check your internet connection and try again.")

    # Set ffmpeg path explicitly for pydub if needed
    ffmpeg_path = os.environ.get("FFMPEG_PATH", "ffmpeg")
    AudioSegment.converter = ffmpeg_path
    
    # Check if ffmpeg is available
    try:
        import subprocess
        result = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"FFmpeg found and working: {ffmpeg_path}")
        else:
            logger.warning(f"FFmpeg found but may have issues: {ffmpeg_path}")
    except Exception as ffmpeg_error:
        logger.warning(f"FFmpeg check failed: {str(ffmpeg_error)}")
        print("⚠️  FFmpeg not found or not working properly.")
        print("   Audio format conversion may not work correctly.")
        print("   Please install FFmpeg and ensure it's in your system PATH.")

except ImportError as import_error:
    logger.error(f"Whisper module not found: {str(import_error)}")
    WHISPER_MODEL = None
    print("❌ Whisper module not installed. Please run: pip install openai-whisper")
except Exception as e:
    logger.error(f"Failed to initialize Whisper: {str(e)}")
    WHISPER_MODEL = None

# Create a directory for temporary file storage
TEMP_DIR = os.path.join(os.getcwd(), "temp_uploads")
os.makedirs(TEMP_DIR, exist_ok=True)


def process_audio_file(file_content, file_ext):
    """Process audio file and return extracted text using Whisper with progress indicators"""
    try:
        start_time = time.time()
        print(f"\nProcessing audio file ({len(file_content)/1024:.1f} KB)...", end="")

        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
        temp_file.write(file_content)
        temp_file.close()

        # Check if Whisper model is available
        if WHISPER_MODEL is None:
            print(" error: Whisper model not available")
            logger.error("Audio transcription failed: Whisper model not available")
            return "[Error: Whisper model not available. Please check server logs.]"

        # Process different audio formats
        if file_ext.lower() in [".wav", ".mp3", ".ogg", ".flac", ".m4a"]:
            try:
                # For formats that may need conversion, use pydub
                if file_ext.lower() not in [".wav", ".mp3", ".flac"]:
                    try:
                        print(" converting format...", end="")
                        logger.info(f"Converting {file_ext} to WAV format for processing")
                        audio = AudioSegment.from_file(temp_file.name, format=file_ext[1:])
                        wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        wav_file.close()
                        audio.export(wav_file.name, format="wav")
                        audio_path = wav_file.name
                    except Exception as conv_error:
                        print(" conversion error!")
                        logger.error(f"Error converting audio format: {str(conv_error)}")
                        return f"[Audio conversion error: {str(conv_error)}. Try uploading a WAV file instead.]"
                else:
                    audio_path = temp_file.name

                # Use Whisper to transcribe
                print(" transcribing...", end="")
                logger.info(f"Transcribing audio file with Whisper: {audio_path}")
                result = WHISPER_MODEL.transcribe(audio_path)
                transcription = result["text"]

                # Log performance
                processing_time = time.time() - start_time
                print(f" complete! ({processing_time:.2f}s) ✓")
                logger.info(f"Audio transcription completed in {processing_time:.2f} seconds")

                # Clean up temporary files
                try:
                    os.unlink(temp_file.name)
                    if audio_path != temp_file.name:
                        os.unlink(audio_path)
                except Exception as clean_error:
                    logger.warning(f"Error cleaning up temp files: {str(clean_error)}")

                return transcription

            except Exception as e:
                print(" error!")
                logger.error(f"Error in audio transcription: {str(e)}")
                return f"[Audio transcription error: {str(e)}]"
        else:
            print(f" error: unsupported format {file_ext}")
            return f"[Unsupported audio format: {file_ext}]"
    except Exception as e:
        print(" error!")
        logger.error(f"Error processing audio: {str(e)}")
        return f"[Error processing audio file: {str(e)}]"


def extract_audio_features(file_path):
    """Extract audio features for analysis"""
    try:
        # Load the audio file using pydub for feature extraction
        audio = AudioSegment.from_file(file_path)

        # Extract basic features
        return {
            "duration_seconds": len(audio) / 1000,  # pydub uses milliseconds
            "channels": audio.channels,
            "sample_rate": audio.frame_rate,
            "bitrate": audio.frame_width * 8 * audio.frame_rate,
        }
    except Exception as e:
        logger.error(f"Error extracting audio features: {str(e)}")
        return {"error": str(e)}


def text_to_speech(
    text: str, output_format: str = "mp3", language: str = "en", voice: str = "default"
):
    """Convert text to speech using gTTS and return audio data"""
    try:
        # Show a simple progress indicator
        print(f"\nGenerating speech ({len(text)} chars)...", end="")
        start_time = time.time()

        if not TTS_AVAILABLE:
            raise Exception("Text-to-speech is not available")

        # Create a temporary file for the audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{output_format}")
        temp_file.close()

        # Use appropriate language code for gTTS
        gtts_lang = language
        if language not in ["en", "fr", "es", "de", "it", "zh", "ja", "ru"]:
            logger.warning(
                f"Language {language} might not be supported by gTTS, falling back to English"
            )
            gtts_lang = "en"

        print(".", end="")  # Progress indicator

        # Generate speech with gTTS
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        tts.save(temp_file.name)

        print(".", end="")  # Progress indicator

        # Read the generated audio file
        with open(temp_file.name, "rb") as f:
            audio_data = f.read()

        # Clean up
        try:
            os.unlink(temp_file.name)
        except Exception as e:
            logger.warning(f"Error removing temporary file: {str(e)}")

        # Log performance
        processing_time = time.time() - start_time
        logger.info(f"TTS completed in {processing_time:.2f} seconds")

        # Show completion
        print(f" complete! ({processing_time:.2f}s, {len(audio_data)/1024:.1f} KB) ✓")

        # If user requested wav but gTTS only does mp3, note that in the result
        actual_format = "mp3" if output_format != "mp3" else output_format
        if actual_format != output_format:
            logger.warning(
                f"Requested format {output_format} not supported by gTTS, using {actual_format} instead"
            )

        return {
            "audio_data": audio_data,
            "format": actual_format,
            "sample_rate": 22050,  # Default for gTTS
            "duration": 0,  # We can't easily determine duration without additional libraries
            "channels": 1,
        }
    except Exception as e:
        print(" error!")
        logger.error(f"Error in text_to_speech: {str(e)}")
        raise


def process_file(file: UploadFile) -> dict:
    """Process audio files and return their content/metadata"""
    file_content = file.file.read()
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext in [".mp3", ".wav", ".ogg", ".flac", ".m4a"]:
        try:
            # Save the audio temporarily
            temp_file_id = str(uuid.uuid4())
            temp_file_path = os.path.join(TEMP_DIR, f"{temp_file_id}{file_ext}")
            with open(temp_file_path, "wb") as f:
                f.write(file_content)

            # Extract text from audio using Whisper
            logger.info(f"Processing audio file: {file.filename}")
            transcribed_text = process_audio_file(file_content, file_ext)

            # Get audio features
            audio_features = extract_audio_features(temp_file_path)

            return {
                "type": "audio",
                "filename": file.filename,
                "size": len(file_content),
                "temp_path": temp_file_path,
                "transcription": transcribed_text,
                "duration": audio_features.get("duration_seconds", "unknown"),
                "sample_rate": audio_features.get("sample_rate", "unknown"),
                "temp_id": temp_file_id,
            }
        except Exception as e:
            logger.error(f"Error processing audio file {file.filename}: {str(e)}")
            return {"type": "error", "filename": file.filename, "error": str(e)}
    else:
        return {"type": "error", "filename": file.filename, "error": "Unsupported file format"}


# Speech-to-Speech function (combines speech-to-text and text-to-speech)
def speech_to_speech(
    audio_file_content: bytes,
    file_ext: str,
    language: str = "en",
    output_format: str = "mp3"
) -> dict:
    """
    Process audio input and return audio response (speech-to-speech)
    This combines speech-to-text and text-to-speech functionality
    """
    try:
        # Step 1: Transcribe audio to text
        transcription = process_audio_file(audio_file_content, file_ext)
        
        if transcription.startswith("[Error:"):
            return {
                "status": "error",
                "message": transcription,
                "step": "transcription"
            }
        
        # Step 2: Convert text back to speech
        audio_result = text_to_speech(transcription, output_format, language)
        
        return {
            "status": "success",
            "transcription": transcription,
            "audio_data": audio_result["audio_data"],
            "format": audio_result["format"],
            "language": language,
            "step": "speech_to_speech"
        }
        
    except Exception as e:
        logger.error(f"Error in speech_to_speech: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "step": "unknown"
        }


# Helper function to check service availability
def check_speech_services():
    """Check the availability of speech processing services"""
    return {
        "whisper_available": WHISPER_MODEL is not None,
        "tts_available": TTS_AVAILABLE,
        "services": {
            "speech_to_text": WHISPER_MODEL is not None,
            "text_to_speech": TTS_AVAILABLE,
            "speech_to_speech": WHISPER_MODEL is not None and TTS_AVAILABLE
        }
    }


# Cleanup function for temporary files
def cleanup_temp_files():
    """Clean up old temporary files"""
    try:
        # Get current time
        now = time.time()

        # Remove files older than 24 hours
        if os.path.exists(TEMP_DIR):
            for filename in os.listdir(TEMP_DIR):
                file_path = os.path.join(TEMP_DIR, filename)
                if os.path.isfile(file_path):
                    # If file is older than 24 hours, delete it
                    if now - os.path.getctime(file_path) > 24 * 3600:
                        try:
                            os.remove(file_path)
                            logger.info(f"Cleaned up old temp file: {filename}")
                        except Exception as e:
                            logger.error(f"Error deleting temp file: {str(e)}")
    except Exception as e:
        logger.error(f"Error in temp file cleanup: {str(e)}")


# Export the main functions for use in other modules
__all__ = [
    "process_audio_file",
    "extract_audio_features", 
    "text_to_speech",
    "speech_to_speech",
    "process_file",
    "check_speech_services",
    "cleanup_temp_files",
    "TTS_AVAILABLE",
    "WHISPER_MODEL"
]
