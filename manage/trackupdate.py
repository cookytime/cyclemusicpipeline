#!/usr/bin/env python3
import os
import sys
from pathlib import Path


def _reexec_with_venv() -> None:
    if __name__ != "__main__":
        return
    if os.environ.get("VIRTUAL_ENV"):
        return
    start = Path(__file__).resolve().parent
    venv_python = None
    for parent in (start, *start.parents):
        candidate = parent / ".venv" / "bin" / "python"
        if candidate.exists():
            venv_python = candidate
            break
    if venv_python is None:
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_reexec_with_venv()

"""
Track management script for Base44 API.
Updates existing tracks or creates new ones based on Spotify ID.
"""

import json
import os
import sys
from pathlib import Path

import requests

# API Configuration
API_BASE_URL = "https://app.base44.com/api"
APP_ID = os.getenv("BASE44_APP_ID", "69668795c37a96600dabcc5c")
API_KEY = os.getenv("BASE44_API_KEY", "812cc3cbff9a402a9695634f8578398b")
ENTITY_TYPE = "Track"


def make_api_request(api_path, method="GET", data=None, params=None):
    """Make authenticated API request to Base44."""
    url = f"{API_BASE_URL}/{api_path}"
    headers = {"api_key": API_KEY, "Content-Type": "application/json"}

    if method.upper() == "GET":
        response = requests.get(url, headers=headers, params=params)
    else:
        response = requests.request(method, url, headers=headers, json=data)

    response.raise_for_status()
    return response.json()


def find_track_by_spotify_id(spotify_id):
    """
    Server-side search for an existing track by Spotify ID.
    Returns the first matching entity dict if found, None otherwise.

    Base44 supports filtering on filterable fields via query parameters
    on the list endpoint (e.g., ?spotify_id=...).
    """
    try:
        normalized_search_id = str(spotify_id).strip()
        print(f"  → Searching (server-side) for spotify_id: {normalized_search_id}")

        # Ask Base44 to filter on the server (no full-table scan)
        entities = make_api_request(
            f"apps/{APP_ID}/entities/{ENTITY_TYPE}",
            params={"spotify_id": normalized_search_id},
        )

        # Some APIs return {"items": [...]} or {"data": [...]}; handle both.
        if isinstance(entities, dict):
            for k in ("items", "data", "results", ENTITY_TYPE.lower(), "entities"):
                if k in entities and isinstance(entities[k], list):
                    entities = entities[k]
                    break

        if isinstance(entities, list) and entities:
            entity = entities[0]
            entity_id = entity.get("id") or entity.get("_id")
            print(
                f"  ✓ Found track. Entity ID: {entity_id} Title: {entity.get('title', 'N/A')}"
            )
            return entity

        print(f"  ✗ No existing track found with spotify_id: '{normalized_search_id}'")
        return None

    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error searching for track: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response body: {e.response.text}")
        return None


def create_track(track_data):
    """Create a new track entity."""
    try:
        print(f"\n{'='*60}")
        print("Creating new track...")
        print(f"{'='*60}")
        print(json.dumps(track_data, indent=2))

        response = requests.post(
            f"{API_BASE_URL}/apps/{APP_ID}/entities/{ENTITY_TYPE}",
            headers={"api_key": API_KEY, "Content-Type": "application/json"},
            json=track_data,
        )

        # Print response details for debugging
        if response.status_code not in [200, 201]:
            print(f"\nAPI Response Status: {response.status_code}")
            print(f"API Response Body: {response.text}")

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error creating track: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response body: {e.response.text}")
        raise


def update_track(entity_id, update_data):
    """Update an existing track entity."""
    try:
        print(f"\n{'='*60}")
        print(f"Updating track entity: {entity_id}")
        print(f"{'='*60}")
        print(json.dumps(update_data, indent=2))

        response = requests.put(
            f"{API_BASE_URL}/apps/{APP_ID}/entities/{ENTITY_TYPE}/{entity_id}",
            headers={"api_key": API_KEY, "Content-Type": "application/json"},
            json=update_data,
        )

        # Print response details for debugging
        if response.status_code != 200:
            print(f"\nAPI Response Status: {response.status_code}")
            print(f"API Response Body: {response.text}")

        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error updating track: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response body: {e.response.text}")
        raise


def load_track_data(spotify_id, captures_dir="./captures"):
    """
    Load track data from metadata and choreography JSON files.
    Expected files:
    - {spotify_id}.metadata.json
    - {spotify_id}.choreography.json
    """
    captures_path = Path(captures_dir)
    metadata_path = captures_path / f"{spotify_id}.metadata.json"
    choreography_path = captures_path / f"{spotify_id}.choreography.json"

    track_data = {}

    # Load Spotify metadata
    if metadata_path.exists():
        print(f"  → Loading metadata from: {metadata_path}")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Map metadata to Base44 fields
        track_data["spotify_id"] = metadata.get("spotify_id", spotify_id)
        track_data["title"] = metadata.get("name", "Unknown Title")

        # Combine artist names
        if metadata.get("artists"):
            track_data["artist"] = ", ".join(
                a.get("name", "") for a in metadata["artists"] if a.get("name")
            )

        # Album info
        if metadata.get("album"):
            track_data["album"] = metadata["album"].get("name")
            if metadata["album"].get("images") and len(metadata["album"]["images"]) > 0:
                track_data["spotify_album_art"] = metadata["album"]["images"][0].get(
                    "url"
                )

        # Duration in minutes
        if metadata.get("duration_ms"):
            track_data["duration_minutes"] = round(metadata["duration_ms"] / 60000, 2)

        # Spotify URL
        if metadata.get("external_urls", {}).get("spotify"):
            track_data["spotify_url"] = metadata["external_urls"]["spotify"]
    else:
        print(f"  ⚠ Metadata file not found: {metadata_path}")

    # Load choreography data
    if choreography_path.exists():
        print(f"  → Loading choreography from: {choreography_path}")
        with open(choreography_path, "r") as f:
            choreography_json = json.load(f)

        # Handle nested "track" structure from your choreography file
        if "track" in choreography_json:
            choreo = choreography_json["track"]
        else:
            choreo = choreography_json

        # Override/add fields from choreography (takes precedence)
        track_data["title"] = choreo.get("title", track_data.get("title", "Unknown"))
        track_data["artist"] = choreo.get("artist", track_data.get("artist", "Unknown"))
        track_data["album"] = choreo.get("album", track_data.get("album"))
        track_data["spotify_id"] = choreo.get("spotify_id", spotify_id)
        track_data["spotify_album_art"] = choreo.get(
            "spotify_album_art", track_data.get("spotify_album_art")
        )
        track_data["spotify_url"] = choreo.get(
            "spotify_url", track_data.get("spotify_url")
        )
        track_data["duration_minutes"] = choreo.get(
            "duration_minutes", track_data.get("duration_minutes")
        )

        # Choreography-specific fields
        track_data["bpm"] = choreo.get("bpm")
        track_data["intensity"] = choreo.get("intensity")
        track_data["focus_area"] = choreo.get("focus_area")
        track_data["track_type"] = choreo.get("track_type")
        track_data["position"] = choreo.get("position")
        track_data["resistance_min"] = choreo.get("resistance_min")
        track_data["resistance_max"] = choreo.get("resistance_max")
        track_data["cadence_min"] = choreo.get("cadence_min")
        track_data["cadence_max"] = choreo.get("cadence_max")

        # Store choreography moves as array (not JSON string)
        if choreo.get("choreography"):
            track_data["choreography"] = choreo["choreography"]

        # Cues as array (not JSON string)
        if choreo.get("cues"):
            track_data["cues"] = choreo["cues"]

        # Notes
        track_data["notes"] = choreo.get("notes", "")
    else:
        print(f"  ⚠ Choreography file not found: {choreography_path}")

    # Remove None values and empty strings to avoid API validation issues
    track_data = {k: v for k, v in track_data.items() if v is not None and v != ""}

    return track_data


def upsert_track(spotify_id, captures_dir="./captures"):
    """
    Check if track exists, then update or create accordingly.
    Returns (entity, was_created) tuple.
    """
    print(f"\n{'='*60}")
    print(f"Processing track: {spotify_id}")
    print(f"{'='*60}\n")

    # Load track data from files
    print("Step 1: Loading track data from files")
    track_data = load_track_data(spotify_id, captures_dir)

    if not track_data:
        print(f"  ✗ No data found for Spotify ID {spotify_id}")
        return None, False

    print(
        f"  ✓ Loaded data for: {track_data.get('title', 'Unknown')} by {track_data.get('artist', 'Unknown')}"
    )
    print(f"     spotify_id in data: '{track_data.get('spotify_id')}'\n")

    # Check if track exists
    print("Step 2: Checking if track already exists in Base44")
    existing_track = find_track_by_spotify_id(spotify_id)

    entity_id = (existing_track.get("id") if existing_track else None) or (
        existing_track.get("_id") if existing_track else None
    )
    if existing_track and entity_id:
        # Update existing track
        entity_id = entity_id
        print(f"\nStep 3: Updating existing track")
        updated = update_track(entity_id, track_data)
        print(f"\n✓ Track updated successfully!")
        return updated, False
    else:
        # Create new track
        print(f"\nStep 3: Creating new track (no existing track found)")
        created = create_track(track_data)
        print(f"\n✓ Track created successfully!")
        return created, True


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python trackupdate.py <spotify_id> [captures_dir]")
        print("Example: python trackupdate.py 3n3Ppam7vgaVa1iaRUc9Lp ./captures")
        sys.exit(1)

    spotify_id = sys.argv[1]
    captures_dir = sys.argv[2] if len(sys.argv) > 2 else "./captures"

    try:
        entity, was_created = upsert_track(spotify_id, captures_dir)
        if entity:
            action = "created" if was_created else "updated"
            print(f"\n{'='*60}")
            print(f"✅ SUCCESS: Track {action}")
            print(f"{'='*60}")
            print(f"Entity ID: {(entity.get('id') or entity.get('_id'))}")
            print(f"Title: {entity.get('title', 'N/A')}")
            print(f"Artist: {entity.get('artist', 'N/A')}")
            print(f"Spotify ID: {entity.get('spotify_id', 'N/A')}")
        else:
            print("\n❌ Failed to process track")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
