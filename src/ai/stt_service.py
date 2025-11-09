"""
STT (Speech-to-Text) service for Serra AI

Supports two providers out of the box:
  - Google Cloud Speech-to-Text (v1 / v2 standard)
  - OpenAI Whisper API (hosted)

MVP usage (file-based transcription):

    from ai.stt_service import STTService, STTConfig

    cfg = STTConfig(
        provider="google",              # or "whisper"
        language_code="en-IN",
        use_phone_call_model=True,
        enable_automatic_punctuation=True,
    )
    stt = STTService(cfg)
    result = stt.transcribe_file("/path/to/audio.wav")
    print(result.text)

Environment variables expected (depending on provider):
  - GOOGLE_APPLICATION_CREDENTIALS: path to GCP service account JSON
  - OPENAI_API_KEY: for Whisper API

Dependencies (add to requirements.txt as needed):
  - google-cloud-speech>=2.26.0
  - openai>=1.40.0
  - python-dotenv>=1.0.1 (optional, if you use .env)
  - pydub (optional, if you plan to normalize/convert audio)

Notes:
  - This MVP provides file-based transcription for reliability.
  - Streaming transcription stubs are included for future extension.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal, Dict, Any
import os
import logging

# Optional: load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass


Provider = Literal["google", "whisper","whisper-local"]


@dataclass
class WordInfo:
    word: str
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    confidence: Optional[float] = None


@dataclass
class STTResult:
    text: str
    language_code: Optional[str] = None
    confidence: Optional[float] = None  # average confidence if available
    words: Optional[List[WordInfo]] = None
    raw: Optional[Any] = None  # provider response for debugging


@dataclass
class STTConfig:
    provider: Provider = "google"
    language_code: str = "en-IN"
    use_phone_call_model: bool = True
    enable_automatic_punctuation: bool = True
    # Google-specific
    google_use_enhanced: bool = True
    google_model_v1: str = "phone_call"  # v1 model names: default, phone_call, video, latest_long, etc.
    google_model_v2: str = "short"  # placeholder – map to v2 later if you adopt v2 client
    # Whisper-specific
    whisper_model: str = "whisper-1"
    # Common audio hints
    sample_rate_hz: Optional[int] = None  # If known; otherwise let provider detect
    encoding: Optional[str] = None        # e.g., LINEAR16, FLAC, OGG_OPUS, MULAW
    diarization: bool = False
    profanity_filter: bool = False
    speech_context_phrases: Optional[List[str]] = None  # boost specific words (brand name, agent name, etc.)
    timeout_seconds: int = 60


class STTService:
    def __init__(self, config: STTConfig):
        self.cfg = config
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        if self.cfg.provider == "google":
            self._init_google()
        elif self.cfg.provider == "whisper":
            self._init_whisper()
        elif self.cfg.provider == "whisper-local":  # Local Whisper (no API, runs on CPU/GPU)
            self.logger.info("Using local Whisper model — no API key or remote setup required.")
        else:
            raise ValueError(f"Unsupported STT provider: {self.cfg.provider}")

    # -----------------------------
    # Provider initializers
    # -----------------------------
    def _init_google(self) -> None:
        try:
            from google.cloud import speech
        except Exception as e:  # pragma: no cover
            raise ImportError("google-cloud-speech is required for Google STT. pip install google-cloud-speech") from e
        # Just store the module; clients can be (re)created per call if desired
        self._g_speech_mod = speech
        self._g_client = speech.SpeechClient()
        self.logger.info("Google Cloud Speech client initialized")

    def _init_whisper(self) -> None:
        try:
            from openai import OpenAI
        except Exception as e:  # pragma: no cover
            raise ImportError("openai>=1.x is required for Whisper API. pip install openai") from e
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in environment")
        self._openai_client = OpenAI(api_key=api_key)
        self.logger.info("OpenAI client initialized for Whisper API")

    # -----------------------------
    # Public API
    # -----------------------------
    def transcribe_file(self, file_path: str) -> STTResult:
        """Transcribe a local audio file.

        Supports common telephony audio (WAV/PCM 8k/16k, MP3, OGG/OPUS, MULAW).
        Returns a normalized STTResult regardless of provider.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(file_path)

        if self.cfg.provider == "google":
            return self._google_transcribe_file(file_path)
        elif self.cfg.provider == "whisper":
            return self._whisper_transcribe_file(file_path)
        elif self.cfg.provider == "whisper-local":
            return self._whisper_local_transcribe_file(file_path)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported provider: {self.cfg.provider}")

    # Placeholder for future live streaming support
    def transcribe_stream_start(self, *_, **__):  # pragma: no cover - stub
        raise NotImplementedError("Streaming STT not implemented in MVP. Use transcribe_file for now.")

    # -----------------------------
    # Google implementation
    # -----------------------------
    def _google_transcribe_file(self, file_path: str) -> STTResult:
        from google.cloud import speech

        with open(file_path, "rb") as f:
            content = f.read()

        audio = speech.RecognitionAudio(content=content)

        # Recognition config
        model = self.cfg.google_model_v1 if self.cfg.use_phone_call_model else "default"
        enable_auto_punct = self.cfg.enable_automatic_punctuation

        diarization_config = None
        if self.cfg.diarization:
            diarization_config = speech.SpeakerDiarizationConfig(
                enable_speaker_diarization=True,
                min_speaker_count=1,
                max_speaker_count=2,  # typical for calls
            )

        speech_contexts = []
        if self.cfg.speech_context_phrases:
            speech_contexts.append(
                speech.SpeechContext(phrases=self.cfg.speech_context_phrases)
            )

        config = speech.RecognitionConfig(
            language_code=self.cfg.language_code,
            enable_automatic_punctuation=enable_auto_punct,
            use_enhanced=self.cfg.google_use_enhanced,
            model=model,
            profanity_filter=self.cfg.profanity_filter,
            sample_rate_hertz=self.cfg.sample_rate_hz or 0,  # let API auto-detect if 0/None
            encoding=(getattr(speech.RecognitionConfig.AudioEncoding, self.cfg.encoding)
                      if self.cfg.encoding else speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED),
            speech_contexts=speech_contexts or None,
            diarization_config=diarization_config,
            enable_word_time_offsets=True,
        )

        self.logger.info(f"Google STT request: model={model}, lang={self.cfg.language_code}")
        response = self._g_client.recognize(config=config, audio=audio, timeout=self.cfg.timeout_seconds)

        if not response.results:
            return STTResult(text="", language_code=self.cfg.language_code, confidence=None, words=[], raw=response)

        # Concatenate best alternatives; compute average confidence
        all_text: List[str] = []
        word_infos: List[WordInfo] = []
        confidences: List[float] = []

        for res in response.results:
            alt = res.alternatives[0]
            all_text.append(alt.transcript.strip())
            if alt.confidence:
                confidences.append(float(alt.confidence))
            for w in getattr(alt, "words", []) or []:
                word_infos.append(
                    WordInfo(
                        word=w.word,
                        start_ms=int(w.start_time.total_seconds() * 1000) if w.start_time else None,
                        end_ms=int(w.end_time.total_seconds() * 1000) if w.end_time else None,
                        confidence=float(getattr(w, "confidence", 0.0)) if hasattr(w, "confidence") else None,
                    )
                )

        text = " ".join(t for t in all_text if t).strip()
        avg_conf = sum(confidences) / len(confidences) if confidences else None

        return STTResult(
            text=text,
            language_code=self.cfg.language_code,
            confidence=avg_conf,
            words=word_infos or None,
            raw=response,
        )

    # -----------------------------
    # Whisper implementation
    # -----------------------------
    def _whisper_transcribe_file(self, file_path: str) -> STTResult:
        # OpenAI Python SDK v1.x
        # Whisper returns just text; word-level timestamps require extra processing and are not included here.
        with open(file_path, "rb") as f:
            transcription = self._openai_client.audio.transcriptions.create(
                model=self.cfg.whisper_model,
                file=f,
                language=self.cfg.language_code.split("-")[0] if self.cfg.language_code else None,
                temperature=0,
                response_format="json",
            )
        text = (transcription.text or "").strip()
        return STTResult(
            text=text,
            language_code=self.cfg.language_code,
            confidence=None,  # Whisper API does not provide confidence
            words=None,
            raw=transcription,
        )
        # -----------------------------
# Whisper Local (Offline) STT
# -----------------------------
    def _whisper_local_transcribe_file(self, file_path: str) -> STTResult:
        try:
            import whisper
        except ImportError:
            raise ImportError("Please install Whisper with: pip install openai-whisper")

    # Load model (tiny, base, small, medium, large)
        model_name = self.cfg.whisper_model or "base"
        self.logger.info(f"Loading Whisper local model: {model_name}")
        model = whisper.load_model(model_name)

        result = model.transcribe(file_path, language=self.cfg.language_code.split('-')[0])

        return STTResult(
            text=result.get("text", "").strip(),
            language_code=self.cfg.language_code,
            confidence=None,
            words=None,
            raw=result
        )


# Convenience factory (optional)
_DEF_PROVIDER = os.getenv("SERRA_STT_PROVIDER", "google").lower()

def make_default_stt(language_code: str = "en-IN") -> STTService:
    cfg = STTConfig(provider=_DEF_PROVIDER, language_code=language_code)
    return STTService(cfg)
