#!/usr/bin/env python3
"""
Main entry point - orchestrates track analysis and Base44 updates.
Processes all WAV files in the captures directory.
"""

import re
import subprocess
import sys
from pathlib import Path


def extract_spotify_id(filename: str) -> str | None:
    """Extract Spotify ID from filename (22-character base62 string)."""
    match = re.search(r"\b([0-9A-Za-z]{22})\b", filename)
    return match.group(1) if match else None


def process_wav_file(wav_path: Path, project_root: Path) -> bool:
    """
    Process a single WAV file: analyze → update Base44.
    Returns True if successful, False otherwise.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {wav_path.name}")
    print(f"{'='*60}\n")

    analyze_script = project_root / "analyze" / "analyze_track.py"
    trackupdate_script = project_root / "manage" / "trackupdate.py"

    # Step 1: Analyze the track (generates music_map.json and choreography.json)
    print("Step 1: Analyzing audio and generating choreography...")
    try:
        result = subprocess.run(
            [sys.executable, str(analyze_script), str(wav_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"❌ Analysis failed: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False

    # Step 2: Extract Spotify ID and update Base44
    spotify_id = extract_spotify_id(wav_path.name)

    if not spotify_id:
        print("⚠️  No Spotify ID found in filename, skipping Base44 update")
        return True  # Analysis succeeded, just no update

    print(f"\nStep 2: Updating Base44 (Spotify ID: {spotify_id})...")
    captures_dir = wav_path.parent

    try:
        result = subprocess.run(
            [sys.executable, str(trackupdate_script), spotify_id, str(captures_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Base44 update failed: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False


def main():
    """Main function - process all WAV files in captures directory."""
    project_root = Path(__file__).parent
    captures_dir = project_root / "captures"

    if not captures_dir.exists():
        print(f"❌ Captures directory not found: {captures_dir}")
        sys.exit(1)

    # Find all WAV files
    wav_files = sorted(captures_dir.glob("*.wav"))

    if not wav_files:
        print(f"No WAV files found in {captures_dir}")
        sys.exit(0)

    print(f"Found {len(wav_files)} WAV file(s) to process\n")

    # Process each file
    results = {"success": 0, "failed": 0, "skipped": 0}

    for wav_path in wav_files:
        # Check if already processed (has choreography.json)
        base_path = wav_path.with_suffix("")
        choreography_path = Path(f"{base_path}.choreography.json")

        if choreography_path.exists():
            print(f"⏭️  Skipping {wav_path.name} (already processed)")
            results["skipped"] += 1
            continue

        # Process the file
        success = process_wav_file(wav_path, project_root)

        if success:
            results["success"] += 1
        else:
            results["failed"] += 1

    # Summary
    print(f"\n{'='*60}")
    print("Processing Complete")
    print(f"{'='*60}")
    print(f"✅ Successful: {results['success']}")
    print(f"❌ Failed: {results['failed']}")
    print(f"⏭️  Skipped: {results['skipped']}")
    print(f"Total: {len(wav_files)}")


if __name__ == "__main__":
    main()
