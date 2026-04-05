from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from services.nearby_places_service import get_nearby_hospitals, get_nearby_police

OUTPUT_DIR = ROOT / "output"
OUTPUT_REPORT = OUTPUT_DIR / "overpass_google_validation_report.md"


@dataclass
class SamplePoint:
    name: str
    latitude: float
    longitude: float


SAMPLE_POINTS = [
    SamplePoint("Istanbul-Taksim", 41.0369, 28.9862),
    SamplePoint("Ankara-Kizilay", 39.9208, 32.8541),
    SamplePoint("Izmir-Konak", 38.4192, 27.1287),
]


def _markdown_section(title: str, rows: List[str]) -> str:
    lines = [f"## {title}", ""]
    lines.extend(rows)
    lines.append("")
    return "\n".join(lines)


def _run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_lines: List[str] = [
        "# Overpass Validation Report",
        "",
        "This report validates nearby hospitals and police stations returned by Overpass.",
        "",
    ]

    console_summary: List[Dict[str, Any]] = []

    for point in SAMPLE_POINTS:
        overpass_hospitals = get_nearby_hospitals(point.latitude, point.longitude, limit=5)
        overpass_police = get_nearby_police(point.latitude, point.longitude, limit=5)

        console_summary.append(
            {
                "point": point.name,
                "hospital_count": len(overpass_hospitals),
                "police_count": len(overpass_police),
            }
        )

        rows = [
            f"- Coordinate: `{point.latitude:.4f}, {point.longitude:.4f}`",
            f"- Hospital count: **{len(overpass_hospitals)}**",
            f"- Police count: **{len(overpass_police)}**",
            "",
            "### Overpass Hospitals",
        ]

        if overpass_hospitals:
            for item in overpass_hospitals:
                rows.append(
                    f"- {item.get('name')} | {item.get('distance_meters')}m | "
                    f"{item.get('latitude')}, {item.get('longitude')}"
                )
        else:
            rows.append("- No hospital results in current search radius.")

        rows.append("")
        rows.append("### Overpass Police")
        if overpass_police:
            for item in overpass_police:
                rows.append(
                    f"- {item.get('name')} | {item.get('distance_meters')}m | "
                    f"{item.get('latitude')}, {item.get('longitude')}"
                )
        else:
            rows.append("- No police results in current search radius.")

        report_lines.append(_markdown_section(point.name, rows))

    OUTPUT_REPORT.write_text("\n".join(report_lines), encoding="utf-8")

    print(json.dumps({"report": str(OUTPUT_REPORT), "summary": console_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _run()
