"""
LLM Response Generator for Serra AI

Purpose
-------
Given user/customer text + conversation context, generate the next agent
utterance and a simple action suggestion (e.g., offer_booking, confirm_booking,
handoff, or end_call).

Design
------
- Uses OpenAI Chat API (any GPT-4/5 style model) via the official SDK v1.x.
- Stateless helper with a thin ConversationState you control from your app.
- Deterministic-ish for compliance steps; creative for pitch (via adjustable temperature).
- Lightweight intent extraction embedded in the same response to avoid extra calls.

Prereqs
-------
Env var: OPENAI_API_KEY must be set (already loaded in main.py via dotenv).

Install: openai>=1.40.0

Example
-------
    from ai.llm_service import LLMConfig, LLMService, ConversationState, ProductInfo

    cfg = LLMConfig(model="gpt-4o-mini", temperature=0.3)
    llm = LLMService(cfg)

    state = ConversationState(
        stage="intro",  # intro | pitch | qualify | schedule | confirm | close
        brand_name="Serra AI",
        company_name="YourCo",
        agent_name="Serra",
        timezone="Asia/Kolkata",
    )

    product = ProductInfo(
        name="Serra AI Caller",
        one_liner="An AI agent that calls leads, explains your offer, and books meetings automatically.",
        key_benefits=["24/7 outreach", "low latency", "calendar booking"],
        pricing_hint="starter plan under $99/mo",
    )

    out = llm.generate_reply(
        user_text="yeah tell me more",
        state=state,
        product=product,
        customer_profile={"name": "Ravi", "company": "Ravi Traders"},
    )

    print(out.text, out.intent, out.action)
"""
from __future__ import annotations

import json
import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal

# -----------------------------
# Data types
# -----------------------------
Intent = Literal[
    "unknown", "greet", "ask_info", "objection",
    "not_interested", "schedule_request", "share_availability",
    "confirm", "reschedule", "goodbye"
]

Action = Literal[
    "none", "offer_pitch", "answer_question", "handle_objection",
    "offer_booking", "collect_timeslot", "create_calendar_event",
    "confirm_booking", "ack_and_close", "handoff_to_human"
]


@dataclass
class ConversationState:
    stage: Literal["intro", "pitch", "qualify", "schedule", "confirm", "close"] = "intro"
    brand_name: str = "Serra AI"
    company_name: str = ""
    agent_name: str = "Serra"
    timezone: str = "Asia/Kolkata"
    language_code: str = "en-IN"
    last_user_message: Optional[str] = None
    summary: str = ""
    lead_score: Optional[int] = None


@dataclass
class ProductInfo:
    name: str
    one_liner: str
    key_benefits: List[str] = field(default_factory=list)
    pricing_hint: Optional[str] = None
    faq: Dict[str, str] = field(default_factory=dict)


@dataclass
class LLMConfig:
    model: str = "gpt-4o-mini"  # change to "gemini-1.5-flash" for Gemini
    temperature: float = 0.3
    max_tokens: int = 250
    json_mode: bool = True


@dataclass
class LLMOutput:
    text: str
    intent: Intent = "unknown"
    action: Action = "none"
    slots: Dict[str, Any] = field(default_factory=dict)
    raw: Optional[Dict[str, Any]] = None


# -----------------------------
# Service
# -----------------------------
class LLMService:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Detect provider (Gemini vs OpenAI)
        self.use_gemini = self.cfg.model.startswith("gemini")

        if self.use_gemini:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError("Install Gemini SDK: pip install google-generativeai")

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise EnvironmentError("GEMINI_API_KEY not set in .env")
            genai.configure(api_key=api_key)
            self.gemini_client = genai
            self.logger.info(f"Using Gemini model: {self.cfg.model}")
        else:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("Install OpenAI SDK: pip install openai>=1.40.0")

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY not set in .env")
            self.client = OpenAI(api_key=api_key)
            self.logger.info(f"Using OpenAI model: {self.cfg.model}")

    def generate_reply(
        self,
        user_text: str,
        state: ConversationState,
        product: ProductInfo,
        customer_profile: Optional[Dict[str, Any]] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> LLMOutput:
        """Generate next reply with intent + action"""
        sys_prompt = self._build_system_prompt(state, product)
        user_payload = self._build_user_payload(user_text, state, product, customer_profile, extra_context)

        # ---------- GEMINI ----------
        if self.use_gemini:
            model = self.gemini_client.GenerativeModel(
                model_name=self.cfg.model,
                system_instruction=sys_prompt
            )
            response = model.generate_content([
                
                {"role": "user", "parts": user_payload}
            ])
            content = (response.text or "{}").strip()
            try:
                data = json.loads(content)
            except Exception:
                repair = model.generate_content([
                    {"role": "user", "parts": [f"Please reformat the previous reply strictly as a single JSON object only, no extra text. Context:\n{user_payload}"]}
                ])
                try:
                    data = json.loads((repair.text or "{}").strip())
                except Exception:
                    data = {"text": content}

        # ---------- OPENAI GPT ----------
        else:
            response = self.client.chat.completions.create(
                model=self.cfg.model,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
                response_format={"type": "json_object"} if self.cfg.json_mode else None,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_payload},
                ],
            )
            content = response.choices[0].message.content or "{}"
            try:
                data = json.loads(content)
            except Exception:
                data = {"text": content}

        # Parse output
        text = (data.get("text") or "").strip()
        intent: Intent = data.get("intent", "unknown")
        action: Action = data.get("action", "none")
        slots = data.get("slots", {}) if isinstance(data.get("slots"), dict) else {}

        return LLMOutput(text=text, intent=intent, action=action, slots=slots, raw=data)

    # -----------------------------
    # Prompt Builders
    # -----------------------------
    def _build_system_prompt(self, state: ConversationState, product: ProductInfo) -> str:
        return f"""
You are {state.agent_name}, a friendly AI voice agent from {state.company_name or state.brand_name},
promoting the product {product.name}.

Your job:
- Explain clearly in 1–2 sentences
- Be polite, human-like
- Try to book an appointment if user is interested
- If user declines, thank them and end politely

Output only JSON with:
"text", "intent", "action", "slots"
"""

    def _build_user_payload(
        self,
        user_text: str,
        state: ConversationState,
        product: ProductInfo,
        customer_profile: Optional[Dict[str, Any]],
        extra_context: Optional[Dict[str, Any]],
    ) -> str:
        profile = json.dumps(customer_profile or {})
        ctx = json.dumps(extra_context or {})
        benefits = " | ".join(product.key_benefits)

        return f"""
<conversation>
stage: {state.stage}
user_text: {user_text}
</conversation>

<product>
name: {product.name}
one_liner: {product.one_liner}
benefits: {benefits}
pricing_hint: {product.pricing_hint or "n/a"}
</product>

<customer_profile>{profile}</customer_profile>
<context>{ctx}</context>

Respond in strict JSON format only.
"""