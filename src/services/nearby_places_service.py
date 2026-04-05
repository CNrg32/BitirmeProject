from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
]
_OVERPASS_ENDPOINTS = [
    endpoint.strip()
    for endpoint in os.environ.get("OVERPASS_API_URLS", ",".join(_DEFAULT_OVERPASS_ENDPOINTS)).split(",")
    if endpoint.strip()
]
_OVERPASS_TIMEOUT_SECONDS = float(os.environ.get("OVERPASS_TIMEOUT_SECONDS", "8.0"))
_OVERPASS_RETRIES = int(os.environ.get("OVERPASS_RETRIES", "2"))
_CACHE_TTL_SECONDS = int(os.environ.get("NEARBY_CACHE_TTL_SECONDS", "180"))
_SEARCH_RADII_METERS = (5000, 10000)

_CACHE: Dict[Tuple[float, float, str, int], Tuple[float, List[Dict[str, Any]]]] = {}


def get_nearby_places(
    latitude: float,
    longitude: float,
    preferred_type: Optional[str] = None,
    limit_per_type: int = 5,
) -> List[Dict[str, Any]]:
    hospitals = get_nearby_hospitals(latitude, longitude, limit=limit_per_type)
    police = get_nearby_police(latitude, longitude, limit=limit_per_type)

    ordered_types = _ordered_place_types(preferred_type)
    grouped = {
        "hospital": hospitals,
        "police": police,
    }

    merged: List[Dict[str, Any]] = []
    for place_type in ordered_types:
        merged.extend(grouped.get(place_type, []))
    return merged


def get_nearby_hospitals(latitude: float, longitude: float, limit: int = 5) -> List[Dict[str, Any]]:
    return _get_places_for_type(latitude, longitude, "hospital", limit)


def get_nearby_police(latitude: float, longitude: float, limit: int = 5) -> List[Dict[str, Any]]:
    return _get_places_for_type(latitude, longitude, "police", limit)


def _get_places_for_type(
    latitude: float,
    longitude: float,
    place_type: str,
    limit: int,
) -> List[Dict[str, Any]]:
    cache_key = (round(latitude, 4), round(longitude, 4), place_type, limit)
    cached = _CACHE.get(cache_key)
    if cached and (time.time() - cached[0]) < _CACHE_TTL_SECONDS:
        return list(cached[1])

    places: List[Dict[str, Any]] = []
    for radius_meters in _SEARCH_RADII_METERS:
        elements = _fetch_overpass_elements(latitude, longitude, place_type, radius_meters)
        places = _normalize_places(elements, latitude, longitude, place_type, limit)
        if len(places) >= limit:
            break

    _CACHE[cache_key] = (time.time(), places)
    return list(places)


def _fetch_overpass_elements(
    latitude: float,
    longitude: float,
    place_type: str,
    radius_meters: int,
) -> List[Dict[str, Any]]:
    amenity = _amenity_for_type(place_type)
    query = f"""
[out:json][timeout:4];
(
  node[\"amenity\"=\"{amenity}\"](around:{radius_meters},{latitude},{longitude});
  way[\"amenity\"=\"{amenity}\"](around:{radius_meters},{latitude},{longitude});
  relation[\"amenity\"=\"{amenity}\"](around:{radius_meters},{latitude},{longitude});
);
out center tags;
""".strip()

    try:
        for endpoint in _OVERPASS_ENDPOINTS:
            for _ in range(_OVERPASS_RETRIES):
                try:
                    response = httpx.post(
                        endpoint,
                        content=query,
                        headers={"Content-Type": "text/plain; charset=utf-8"},
                        timeout=_OVERPASS_TIMEOUT_SECONDS,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    elements = payload.get("elements", [])
                    if isinstance(elements, list):
                        return elements
                except Exception:
                    continue
    except Exception as exc:
        logger.warning(
            "Nearby places lookup failed for %s at %.5f, %.5f: %s",
            place_type,
            latitude,
            longitude,
            exc,
        )
    logger.warning(
        "Nearby places lookup failed for %s at %.5f, %.5f across all Overpass endpoints",
        place_type,
        latitude,
        longitude,
    )
    return []


def _normalize_places(
    elements: List[Dict[str, Any]],
    latitude: float,
    longitude: float,
    place_type: str,
    limit: int,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen: set[Tuple[str, float, float]] = set()

    for element in elements:
        tags = element.get("tags") or {}
        place_lat = element.get("lat")
        place_lon = element.get("lon")

        center = element.get("center") or {}
        if place_lat is None:
            place_lat = center.get("lat")
        if place_lon is None:
            place_lon = center.get("lon")

        if place_lat is None or place_lon is None:
            continue

        name = str(tags.get("name") or _fallback_name(place_type)).strip()
        dedupe_key = (name.lower(), round(float(place_lat), 5), round(float(place_lon), 5))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        distance_meters = round(_haversine_meters(latitude, longitude, float(place_lat), float(place_lon)))
        normalized.append(
            {
                "id": f"osm:{element.get('type', 'node')}:{element.get('id', 'unknown')}",
                "type": place_type,
                "name": name,
                "address": _build_address(tags),
                "distance_meters": distance_meters,
                "latitude": float(place_lat),
                "longitude": float(place_lon),
                "phone": tags.get("phone") or tags.get("contact:phone"),
                "eta_minutes": max(1, round(distance_meters / 800)) if distance_meters else 1,
            }
        )

    normalized.sort(key=lambda item: item["distance_meters"])
    return normalized[:limit]


def _ordered_place_types(preferred_type: Optional[str]) -> List[str]:
    normalized_type = (preferred_type or "").strip().lower()
    if normalized_type == "police":
        return ["police", "hospital"]
    return ["hospital", "police"]


def _amenity_for_type(place_type: str) -> str:
    if place_type == "police":
        return "police"
    return "hospital"


def _fallback_name(place_type: str) -> str:
    if place_type == "police":
        return "Police Station"
    return "Hospital"


def _build_address(tags: Dict[str, Any]) -> str:
    parts = [
        tags.get("addr:street"),
        tags.get("addr:housenumber"),
        tags.get("addr:suburb"),
        tags.get("addr:district"),
        tags.get("addr:city"),
    ]
    cleaned = [str(part).strip() for part in parts if part]
    if cleaned:
        return ", ".join(cleaned)
    return str(tags.get("addr:full") or tags.get("addr:place") or "Adres bilgisi yok")


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c