import os
from dotenv import load_dotenv
from ai.stt_service import STTService, STTConfig
from ai.llm_service import LLMService, LLMConfig, ConversationState, ProductInfo
import google.generativeai as genai
import os
# Load environment variables
load_dotenv()

# Ensure Google can find the credentials JSON
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

def transcribe_audio():
    config = STTConfig(
    provider="whisper-local",  # instead of "google"
    whisper_model="base",
    language_code="en-IN"
    )
    stt = STTService(config)
    result = stt.transcribe_file("test_recordings/test-01.wav")  # Use relative path
    print("\n🎤 Transcription Result:\n", result.text)
def test_gemini():
    config = LLMConfig(
    model="gemini-2.5-flash",  # or gemini-pro
    temperature=0.3
    )

    llm = LLMService(config)

    state = ConversationState(stage="intro", brand_name="Serra AI")
    product = ProductInfo(
        name="Serra AI Agent",
        one_liner="AI that calls leads, explains your service, and books appointments.",
        key_benefits=["Automated outreach", "Calendar booking", "Affordable"]
    )

    response = llm.generate_reply(
        user_text="Can you explain your product?",
        state=state,
        product=product,
        customer_profile={"name": "Ravi", "company": "TechCorp"}
    )

    print(response.text, response.intent, response.action)
if __name__ == "__main__":
    test_gemini()
