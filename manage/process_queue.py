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
Process queued tracks from Base44 sync.
- Analyzes audio files
- Generates choreography via OpenAI
- Updates Base44 with results
- Updates processing playlist
"""

import json
import subprocess
import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
QUEUE_DIR = PROJECT_ROOT / "queue"
CAPTURES_DIR = PROJECT_ROOT / "captures"
ANALYZE_SCRIPT = PROJECT_ROOT / "analyze" / "analyze_track.py"
PLAYLIST_SYNC_SCRIPT = PROJECT_ROOT / "manage" / "manage_processing_playlist.py"


def get_queued_tracks():
    """Get list of tracks in queue directory."""
    if not QUEUE_DIR.exists():
        return []

    queue_files = sorted(QUEUE_DIR.glob("*.json"))
    return queue_files


def process_track(queue_file):
    """
    Process a single queued track:
    1. Check if audio exists
    2. Run analysis + choreography generation
    3. Update Base44
    4. Archive queue file
    """
    # Load queue data
    with open(queue_file, "r", encoding="utf-8") as f:
        track_data = json.load(f)

    spotify_id = track_data.get("spotify_id")
    title = track_data.get("title", "Unknown")
    artist = track_data.get("artist", "Unknown")

    print(f"\nProcessing: {artist} - {title}")
    print(f"  Spotify ID: {spotify_id}")

    # Check if audio exists
    wav_path = CAPTURES_DIR / f"{spotify_id}.wav"
    if not wav_path.exists():
        print(f"  ✗ Audio file missing: {wav_path}")
        print("    Skipping - capture audio first")
        return False

    # Check if already has choreography
    choreo_path = CAPTURES_DIR / f"{spotify_id}.choreography.json"
    if choreo_path.exists():
        print(f"  ⏭  Already has choreography: {choreo_path.name}")
        # Archive the queue file
        archive_queue_file(queue_file)
        return True

    # Run analyze_track.py (does analysis + choreography + Base44 update)
    print("  → Running analysis and choreography generation...")

    try:
        result = subprocess.run(
            [sys.executable, str(ANALYZE_SCRIPT), str(wav_path)],
            check=True,
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.stderr:
            print("  Warnings:", result.stderr)

        print("  ✓ Processing complete!")

        # Archive the queue file
        archive_queue_file(queue_file)
        return True

    except subprocess.CalledProcessError as e:
        print(f"  ✗ Processing failed: {e}")
        if e.stdout:
            print(f"     stdout: {e.stdout[:200]}")
        if e.stderr:
            print(f"     stderr: {e.stderr[:200]}")
        return False


def archive_queue_file(queue_file):
    """Move processed queue file to archive."""
    archive_dir = QUEUE_DIR / "processed"
    archive_dir.mkdir(exist_ok=True)

    archive_path = archive_dir / queue_file.name
    queue_file.rename(archive_path)
    print(f"  ✓ Archived: {archive_path.name}")


def sync_processing_playlist():
    """Update the Spotify processing playlist after processing tracks."""
    if not PLAYLIST_SYNC_SCRIPT.exists():
        print("\n⚠️  Playlist sync script not found - skipping playlist update")
        return False

    print(f"\n{'='*60}")
    print("Updating Processing Playlist")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            [sys.executable, str(PLAYLIST_SYNC_SCRIPT)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Playlist update failed: {e}")
        if e.stdout:
            print(e.stdout)
        return False


def process_queue():
    """Process all tracks in queue."""
    print(f"\n{'='*60}")
    print("Processing Queued Tracks")
    print(f"{'='*60}")

    # Verify analyzer exists
    if not ANALYZE_SCRIPT.exists():
        print(f"✗ Analyzer script not found: {ANALYZE_SCRIPT}")
        sys.exit(1)

    # Get queued tracks
    queue_files = get_queued_tracks()

    if not queue_files:
        print("\n✓ Queue is empty - no tracks to process")
        return

    print(f"\nFound {len(queue_files)} tracks in queue\n")

    stats = {"success": 0, "skipped": 0, "failed": 0}

    for idx, queue_file in enumerate(queue_files, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(queue_files)}] {queue_file.name}")
        print(f"{'='*60}")

        try:
            success = process_track(queue_file)
            if success:
                stats["success"] += 1
            else:
                stats["skipped"] += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            stats["failed"] += 1

    # Summary
    print(f"\n{'='*60}")
    print("Queue Processing Complete")
    print(f"{'='*60}")
    print(f"Total queued: {len(queue_files)}")
    print(f"✓ Successfully processed: {stats['success']}")
    print(f"⏭  Skipped (missing audio): {stats['skipped']}")
    print(f"✗ Failed: {stats['failed']}")

    # Update processing playlist if tracks were processed
    if stats["success"] > 0:
        sync_processing_playlist()


def main():
    """Main entry point."""
    process_queue()


if __name__ == "__main__":
    main()
