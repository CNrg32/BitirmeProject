from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TriageLevel(str, Enum):
    CRITICAL = "CRITICAL"
    URGENT = "URGENT"
    NON_URGENT = "NON_URGENT"


class Category(str, Enum):
    MEDICAL = "medical"
    CRIME = "crime"
    FIRE = "fire"
    OTHER = "other"


class MetaInput(BaseModel):
    deaths: float = 0
    potential_death: float = 0
    false_alarm: float = 0


class SlotsInput(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None
    severity_1_10: Optional[int] = None
    duration_minutes: Optional[int] = None
    location_hint: Optional[str] = None
    red_flags: Optional[List[str]] = None


class PredictRequest(BaseModel):
    text_en: str = Field(..., min_length=1, description="Emergency description in English")
    meta: MetaInput = Field(default_factory=MetaInput)
    slots: SlotsInput = Field(default_factory=SlotsInput)

    model_config = {"json_schema_extra": {
        "examples": [{
            "text_en": "My dad is not breathing and unconscious, he is 63 years old",
            "meta": {"deaths": 0, "potential_death": 0, "false_alarm": 0},
            "slots": {"age": 63, "severity_1_10": 9, "duration_minutes": 10},
        }]
    }}


class ImageClassification(BaseModel):
    detected_class: str = Field(..., description="Predicted emergency class (e.g. Arson, RoadAccidents)")
    confidence: float = Field(..., description="Softmax confidence 0-1")
    top3: List[Dict[str, Any]] = Field(default_factory=list, description="Top-3 predictions")
    dispatch_units: List[str] = Field(default_factory=list, description="Recommended dispatch units")
    mapped_category: str = Field("other", description="Mapped triage category (medical/crime/fire/other)")


class ImageConsistency(BaseModel):
    is_consistent: bool = Field(True, description="Whether image and text agree")
    consistency_score: float = Field(0.5, description="0.0 (contradictory) to 1.0 (fully aligned)")
    consistency_detail: str = Field("", description="CONSISTENT / INCONSISTENT / POSSIBLY FAKE")
    possible_fake: bool = Field(False, description="Image appears unrelated to the reported emergency")
    risk_notes: List[str] = Field(default_factory=list, description="Risk observations")


class ImageAnalysisResult(BaseModel):
    classification: Optional[ImageClassification] = None
    consistency: Optional[ImageConsistency] = None
    summary: str = Field("", description="One-line human-readable summary")
    available: bool = Field(False, description="Whether image analysis was performed")


class PredictResponse(BaseModel):
    category: str
    triage_level: str
    confidence: Optional[float] = None
    red_flags: List[str] = Field(default_factory=list)
    slots: Dict[str, Any] = Field(default_factory=dict)
    needs_more_info: bool = False
    recommended_questions: List[str] = Field(default_factory=list)
    image_analysis: Optional[ImageAnalysisResult] = None
    sentiment_result: Optional[Dict[str, Any]] = Field(None, description="Panic level and sentiment triage from voice/text")


class SessionStartRequest(BaseModel):
    language: Optional[str] = Field(None, description="ISO-639-1 language code. If omitted, language is auto-detected from first voice message.")


class SessionStartResponse(BaseModel):
    session_id: str
    greeting: str
    greeting_audio_url: Optional[str] = None
    greeting_audio_b64: Optional[str] = Field(None, description="Greeting TTS audio as base64 for inline playback")


class SessionTranscribeRequest(BaseModel):
    session_id: str
    audio_base64: str = Field(..., description="Base64-encoded audio to transcribe")


class SessionTranscribeResponse(BaseModel):
    session_id: str
    transcript: str = Field("", description="Transcribed text from audio")
    detected_language: Optional[str] = Field(None, description="Auto-detected language code")


class SessionMessageRequest(BaseModel):
    session_id: str
    text: Optional[str] = None
    audio_base64: Optional[str] = None
    image_base64: Optional[str] = Field(None, description="Optional base64-encoded image for scene analysis")


class SessionMessageResponse(BaseModel):
    session_id: str
    assistant_text: str
    assistant_audio_url: Optional[str] = None
    assistant_audio_b64: Optional[str] = Field(None, description="TTS audio as base64 for playback")
    user_transcript: Optional[str] = Field(None, description="ASR transcript of user voice input")
    triage_result: Optional[PredictResponse] = None
    image_analysis: Optional[ImageAnalysisResult] = None
    report: Optional[str] = None
    is_complete: bool = False
