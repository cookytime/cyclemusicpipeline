#!/usr/bin/env python3

import logging
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cyclemusicpipeline")

"""
Main entry point - orchestrates track analysis and Base44 updates.
Processes all WAV files in the captures directory.
"""

import argparse
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
    logger.info(f"Processing: {wav_path.name}")

    analyze_script = project_root / "analyze" / "analyze_track.py"
    trackupdate_script = project_root / "manage" / "trackupdate.py"

    # Step 1: Analyze the track (generates music_map.json and choreography.json)
    logger.info("Step 1: Analyzing audio and generating choreography...")
    try:
        result = subprocess.run(
            [sys.executable, str(analyze_script), str(wav_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(result.stdout.strip())
        if result.stderr:
            logger.warning(result.stderr.strip())
    except subprocess.CalledProcessError as e:
        logger.error(f"Analysis failed: {e}")
        if e.stdout:
            logger.error(e.stdout.strip())
        if e.stderr:
            logger.error(e.stderr.strip())
        return False
    except Exception as e:
        logger.exception(f"Unexpected error during analysis: {e}")
        return False

    # Step 2: Extract Spotify ID and update Base44
    spotify_id = extract_spotify_id(wav_path.name)

    if not spotify_id:
        logger.warning("No Spotify ID found in filename, skipping Base44 update")
        return True  # Analysis succeeded, just no update

    logger.info(f"Step 2: Updating Base44 (Spotify ID: {spotify_id})...")
    captures_dir = wav_path.parent

    try:
        result = subprocess.run(
            [sys.executable, str(trackupdate_script), spotify_id, str(captures_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(result.stdout.strip())
        if result.stderr:
            logger.warning(result.stderr.strip())
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Base44 update failed: {e}")
        if e.stdout:
            logger.error(e.stdout.strip())
        if e.stderr:
            logger.error(e.stderr.strip())
        return False
    except Exception as e:
        logger.exception(f"Unexpected error during Base44 update: {e}")
        return False


def process_captures() -> None:
    """Process all WAV files in the captures directory."""
    project_root = Path(__file__).parent
    captures_dir = project_root / "captures"

    if not captures_dir.exists():
        logger.error(f"Captures directory not found: {captures_dir}")
        sys.exit(1)

    # Find all WAV files
    wav_files = sorted(captures_dir.glob("*.wav"))

    if not wav_files:
        logger.warning(f"No WAV files found in {captures_dir}")
        sys.exit(0)

    logger.info(f"Found {len(wav_files)} WAV file(s) to process")

    # Process each file
    results = {"success": 0, "failed": 0, "skipped": 0}

    for wav_path in wav_files:
        # Check if already processed (has choreography.json)
        base_path = wav_path.with_suffix("")
        choreography_path = Path(f"{base_path}.choreography.json")

        if choreography_path.exists():
            logger.info(f"Skipping {wav_path.name} (already processed)")
            results["skipped"] += 1
            continue

        # Process the file
        success = process_wav_file(wav_path, project_root)

        if success:
            results["success"] += 1
        else:
            results["failed"] += 1

    # Summary
    logger.info("Processing Complete")
    logger.info(f"Successful: {results['success']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info(f"Skipped: {results['skipped']}")
    logger.info(f"Total: {len(wav_files)}")


def run_processing_playlist_sync() -> None:
    """Sync the Spotify processing playlist."""
    from manage import manage_processing_playlist

    manage_processing_playlist.main()


def run_playlist_capture() -> None:
    """Capture audio for the configured Spotify playlist."""
    from capture import auto_capture_playlist_only

    auto_capture_playlist_only.main()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CycleMusicPipeline orchestration entry point."
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser(
        "process-captures",
        help="Process WAV files in the captures directory (default).",
    )
    subparsers.add_parser(
        "sync-playlist",
        help="Sync the Spotify processing playlist for choreography.",
    )
    subparsers.add_parser(
        "capture-playlist",
        help="Capture audio from the configured Spotify playlist.",
    )
    return parser


def main():
    """Main function - orchestrates pipeline tasks."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "sync-playlist":
        run_processing_playlist_sync()
    elif args.command == "capture-playlist":
        run_playlist_capture()
    else:
        process_captures()


if __name__ == "__main__":
    main()
