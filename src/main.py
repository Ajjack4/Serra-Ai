import os
from dotenv import load_dotenv
from ai.stt_service import STTService, STTConfig

# Load environment variables
load_dotenv()

# Ensure Google can find the credentials JSON
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

def transcribe_audio():
    config = STTConfig(
        provider="google",
        language_code="en-US",
        use_phone_call_model=True,
        enable_automatic_punctuation=True
    )
    stt = STTService(config)
    result = stt.transcribe_file("test_recordings/test-01.wav")  # Use relative path
    print("\n🎤 Transcription Result:\n", result.text)

if __name__ == "__main__":
    print("✅ Environment Variables Loaded.")
    print("🔹 Google Credentials:", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    transcribe_audio()
