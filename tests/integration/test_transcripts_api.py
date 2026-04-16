from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_list_transcripts(app_client_no_lifespan):
    store = MagicMock()
    store.list.return_value = [
        {
            "transcript_id": "abc123",
            "path": "audio/call_1.mp3",
            "text": "Caller needs ambulance.",
            "avg_confidence": 0.91,
            "chunks": "2",
            "source": "auto_transcripts.csv",
            "imported_at": "2026-04-16T00:00:00+00:00",
        }
    ]

    with patch("services.transcript_store.get_transcript_store", return_value=store):
        response = app_client_no_lifespan.get("/transcripts?limit=1")

    assert response.status_code == 200
    assert response.json()["transcripts"][0]["transcript_id"] == "abc123"
    store.list.assert_called_once_with(limit=1)


def test_import_auto_transcripts(app_client_no_lifespan, tmp_path):
    csv_path = tmp_path / "auto_transcripts.csv"
    csv_path.write_text(
        'path,text,avg_confidence,chunks\n'
        'audio/call_1.mp3,"Caller needs ambulance.",0.91,2\n',
        encoding="utf-8",
    )
    store = MagicMock()
    store.table_name = "EmergencyAutoTranscripts"
    store.put_many.return_value = 1

    with patch("services.transcript_store.get_transcript_store", return_value=store):
        response = app_client_no_lifespan.post(
            "/transcripts/import-auto-csv",
            json={"csv_path": str(csv_path)},
        )

    assert response.status_code == 200
    assert response.json() == {
        "imported_count": 1,
        "table_name": "EmergencyAutoTranscripts",
    }
    assert len(store.put_many.call_args.args[0]) == 1
