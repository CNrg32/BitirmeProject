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
    caller_name: Optional[str] = None
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
    latitude: Optional[float] = Field(None, description="GPS latitude from mobile device")
    longitude: Optional[float] = Field(None, description="GPS longitude from mobile device")


class NearbyPlacesRequest(BaseModel):
    latitude: float = Field(..., description="GPS latitude")
    longitude: float = Field(..., description="GPS longitude")
    preferred_type: Optional[str] = Field(None, description="hospital | police")
    limit_per_type: int = Field(5, ge=1, le=10, description="Max results per type")


class NearbyPlacesResponse(BaseModel):
    nearby_places: List[Dict[str, Any]] = Field(default_factory=list)


class TranscriptImportRequest(BaseModel):
    csv_path: Optional[str] = Field(
        None,
        description="Path to CSV file. Defaults to data/labels/auto_transcripts.csv",
    )


class TranscriptImportResponse(BaseModel):
    imported_count: int
    table_name: Optional[str] = None


class TranscriptRecordResponse(BaseModel):
    transcript_id: str
    path: str
    text: str
    avg_confidence: Optional[float] = None
    chunks: Optional[str] = None
    source: Optional[str] = None
    imported_at: Optional[str] = None


class TranscriptListResponse(BaseModel):
    transcripts: List[TranscriptRecordResponse] = Field(default_factory=list)


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
    
    # FAZ 8-9: Dispatch & Resume fields
    chatbot_mode: str = Field("normal", description="normal | fallback (TextAnalyze fail)")
    dispatch_status: str = Field("PENDING", description="PENDING | DISPATCHED | SILENT_DISPATCHED | FALLBACK_PENDING")
    
    summary_card: Optional[Dict[str, Any]] = Field(None, description="""
        Optional summary for post-dispatch state:
        {
            "category": "medical|fire|crime|other",
            "triage_level": "CRITICAL|URGENT|NON_URGENT",
            "dispatch_time": "ISO 8601",
            "dispatch_target": "police|medical|fire",
            "estimated_arrival_min": 5,
            "collected_slots": {...}
        }
    """)
    
    resume_prompt: Optional[str] = Field(None, description="Tutorial/help text if resuming after timeout")
    followup_status: Optional[str] = Field(None, description="waiting_for_update | dispatch_sent | no_dispatch_needed")
    
    # FAZ 9: Nearby places (hospitals, police)
    nearby_places: Optional[List[Dict[str, Any]]] = Field(None, description="""
        Optional ordered list of nearby facilities from OpenStreetMap/Overpass.
        The backend can return both hospitals and police stations in a single list,
        and the client can filter/group them by `type`.
        [{
            "id": "osm:node:12345",
            "type": "hospital|police",
            "name": "Hospital Name",
            "address": "Street, District, City",
            "distance_meters": 1500,
            "latitude": 41.0082,
            "longitude": 28.9784,
            "phone": "+90 212 ...",
            "eta_minutes": 5
        }]
    """)
