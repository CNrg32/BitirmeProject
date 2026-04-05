from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from services import nearby_places_service


def test_get_nearby_hospitals_normalizes_and_limits_results():
    nearby_places_service._CACHE.clear()

    response = MagicMock()
    response.json.return_value = {
        "elements": [
            {
                "id": 1,
                "type": "node",
                "lat": 41.0,
                "lon": 29.0,
                "tags": {"name": "A Hospital", "addr:city": "Istanbul"},
            },
            {
                "id": 2,
                "type": "node",
                "lat": 41.001,
                "lon": 29.001,
                "tags": {"name": "B Hospital", "addr:city": "Istanbul"},
            },
        ]
    }
    response.raise_for_status.return_value = None

    with patch("services.nearby_places_service.httpx.post", return_value=response):
        results = nearby_places_service.get_nearby_hospitals(41.0, 29.0, limit=1)

    assert len(results) == 1
    assert results[0]["type"] == "hospital"
    assert results[0]["name"] == "A Hospital"
    assert results[0]["address"] == "Istanbul"


def test_get_nearby_places_orders_preferred_type_first():
    with patch(
        "services.nearby_places_service.get_nearby_hospitals",
        return_value=[{"id": "h1", "type": "hospital", "name": "Hospital"}],
    ), patch(
        "services.nearby_places_service.get_nearby_police",
        return_value=[{"id": "p1", "type": "police", "name": "Police"}],
    ):
        results = nearby_places_service.get_nearby_places(
            41.0,
            29.0,
            preferred_type="police",
            limit_per_type=5,
        )

    assert [item["type"] for item in results] == ["police", "hospital"]