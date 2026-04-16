from __future__ import annotations

from decimal import Decimal

from services.transcript_store import (
    TranscriptRecord,
    load_auto_transcripts_csv,
    stable_transcript_id,
)


def test_stable_transcript_id_is_deterministic():
    value = "data/raw/911_recordings/audio/call_1.mp3"

    first = stable_transcript_id(value)
    second = stable_transcript_id(value)

    assert first == second
    assert len(first) == 32


def test_transcript_record_from_csv_row():
    record = TranscriptRecord.from_csv_row(
        {
            "path": "audio/call_1.mp3",
            "text": "Caller needs ambulance.",
            "avg_confidence": "0.91",
            "chunks": "2",
        },
        imported_at="2026-04-16T00:00:00+00:00",
    )

    assert record.transcript_id == stable_transcript_id("audio/call_1.mp3")
    assert record.avg_confidence == Decimal("0.91")
    assert record.to_item()["chunks"] == "2"


def test_load_auto_transcripts_csv_handles_multiline_text(tmp_path):
    csv_path = tmp_path / "auto_transcripts.csv"
    csv_path.write_text(
        'path,text,avg_confidence,chunks\n'
        'audio/call_1.mp3,"line one\nline two",0.88,3\n',
        encoding="utf-8",
    )

    records = load_auto_transcripts_csv(csv_path)

    assert len(records) == 1
    assert records[0].text == "line one\nline two"
    assert records[0].avg_confidence == Decimal("0.88")
