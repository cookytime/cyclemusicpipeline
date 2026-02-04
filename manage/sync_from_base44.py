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
Sync tracks from Base44 API.
- Downloads tracks without choreography
- Queues them for analysis
- Generates choreography via OpenAI
- Updates Base44 with results
"""

import json
from pathlib import Path

from manage.base44_utils import filter_tracks_needing_choreography, get_all_tracks
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
QUEUE_DIR = PROJECT_ROOT / "queue"
CAPTURES_DIR = PROJECT_ROOT / "captures"


def save_track_to_queue(track, queue_dir):
    """
    Save track metadata to queue directory for processing.
    Creates: queue/{spotify_id}.json
    """
    spotify_id = track.get("spotify_id")
    queue_file = queue_dir / f"{spotify_id}.json"

    # Save track data
    with open(queue_file, "w", encoding="utf-8") as f:
        json.dump(track, f, indent=2)

    return queue_file


def download_audio_if_needed(track, captures_dir):
    """
    Check if audio file exists for this track.
    Returns True if audio exists, False if needs to be downloaded.

    Note: This assumes you have the audio file named {spotify_id}.wav
    If you don't have the audio, you'll need to capture it from Spotify.
    """
    spotify_id = track.get("spotify_id")
    wav_path = captures_dir / f"{spotify_id}.wav"

    if wav_path.exists() and wav_path.stat().st_size > 100_000:
        print(f"    ✓ Audio exists: {wav_path.name}")
        return True

    print(f"    ✗ Audio missing: {wav_path.name}")
    print("      You need to capture this track from Spotify")
    return False


def create_metadata_from_track(track, captures_dir):
    """
    Create metadata.json file from Base44 track data.
    This allows analyze_track.py to work without Spotify API calls.
    """
    spotify_id = track.get("spotify_id")
    metadata_path = captures_dir / f"{spotify_id}.metadata.json"

    # Convert Base44 track to Spotify metadata format
    metadata = {
        "spotify_id": spotify_id,
        "name": track.get("title", "Unknown"),
        "artists": [{"name": track.get("artist", "Unknown"), "id": None, "uri": None}],
        "album": {
            "name": track.get("album"),
            "id": None,
            "release_date": None,
            "images": (
                [{"url": track.get("spotify_album_art")}]
                if track.get("spotify_album_art")
                else []
            ),
        },
        "duration_ms": int((track.get("duration_minutes") or 0) * 60000),
        "explicit": False,
        "uri": f"spotify:track:{spotify_id}",
        "external_urls": {"spotify": track.get("spotify_url")},
        "isrc": None,
        "synced_from_base44": True,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"    ✓ Metadata created: {metadata_path.name}")
    return metadata_path


def sync_tracks():
    """
    Main sync function:
    1. Get all tracks from Base44
    2. Filter those needing choreography
    3. Queue them for processing
    4. Create metadata files
    """
    # Create directories
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Base44 Track Sync - Find Tracks Needing Choreography")
    print(f"{'='*60}\n")

    # Step 1: Get all tracks
    print("Fetching all tracks from Base44...")
    all_tracks = get_all_tracks()
    if not all_tracks:
        print("No tracks found in Base44")
        return

    print(f"  ✓ Retrieved {len(all_tracks)} tracks")

    # Step 2: Filter tracks needing choreography
    print("\nFiltering tracks without choreography...")
    needs_choreo = filter_tracks_needing_choreography(all_tracks)

    print(f"  ✓ Found {len(needs_choreo)} tracks needing choreography")

    # Step 2.5: Deduplicate tracks, keeping newest and removing oldest
    print("\nDeduplicating tracks...")
    seen = {}
    duplicates_removed = 0

    for track in needs_choreo:
        spotify_id = track.get("spotify_id")
        if spotify_id in seen:
            # Compare timestamps/IDs - keep the one with higher base44_id (newest)
            existing = seen[spotify_id]
            existing_id = existing.get("base44_id", 0)
            current_id = track.get("base44_id", 0)

            if current_id > existing_id:
                # Current is newer, replace the old one
                seen[spotify_id] = track
                duplicates_removed += 1
            else:
                # Existing is newer or same, skip current
                duplicates_removed += 1
        else:
            seen[spotify_id] = track

    needs_choreo = list(seen.values())
    print(f"  ✓ Removed {duplicates_removed} duplicate(s)")
    print(f"  ✓ {len(needs_choreo)} unique tracks remaining\n")

    if not needs_choreo:
        print("✅ All tracks already have choreography!")
        return

    # Step 3: Process each track
    print(f"{'='*60}")
    print(f"Processing {len(needs_choreo)} tracks")
    print(f"{'='*60}\n")

    stats = {"queued": 0, "has_audio": 0, "missing_audio": 0, "errors": 0}

    for idx, track in enumerate(needs_choreo, 1):
        spotify_id = track.get("spotify_id")
        title = track.get("title", "Unknown")
        artist = track.get("artist", "Unknown")

        print(f"[{idx}/{len(needs_choreo)}] {artist} - {title}")
        print(f"  Spotify ID: {spotify_id}")

        try:
            # Save to queue
            queue_file = save_track_to_queue(track, QUEUE_DIR)
            print(f"    ✓ Queued: {queue_file.name}")
            stats["queued"] += 1

            # Create metadata file
            create_metadata_from_track(track, CAPTURES_DIR)

            # Check if audio exists
            if download_audio_if_needed(track, CAPTURES_DIR):
                stats["has_audio"] += 1
            else:
                stats["missing_audio"] += 1

            print()

        except Exception as e:
            print(f"    ✗ Error: {e}\n")
            stats["errors"] += 1

    # Summary
    print(f"\n{'='*60}")
    print("Sync Complete")
    print(f"{'='*60}")
    print(f"Total tracks needing choreography: {len(needs_choreo)}")
    print(f"✓ Queued for processing: {stats['queued']}")
    print(f"✓ Have audio files: {stats['has_audio']}")
    print(f"⚠ Missing audio files: {stats['missing_audio']}")
    print(f"✗ Errors: {stats['errors']}")

    if stats["missing_audio"] > 0:
        print(f"\n⚠️  {stats['missing_audio']} tracks need audio capture from Spotify")
        print("   Run: python capture/auto_capture_playlist_only.py")

    if stats["has_audio"] > 0:
        print(f"\n✅ Ready to process {stats['has_audio']} tracks with existing audio")
        print("   Run: python manage/process_queue.py")


def main():
    """Main entry point."""
    sync_tracks()


if __name__ == "__main__":
    main()
