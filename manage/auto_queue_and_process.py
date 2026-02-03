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
Automate queue creation for tracks needing choreography.
- Fetches tracks from Base44 needing choreography
- Creates a queue .json file for each track in queue/
- Then processes the queue (analysis, choreography, Base44 update)
"""


import os
import sys
import json
from pathlib import Path

# Ensure parent directory is in sys.path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

from base44_utils import get_all_tracks, filter_tracks_needing_choreography
from process_queue import process_queue

PROJECT_ROOT = Path(__file__).parent.parent
QUEUE_DIR = PROJECT_ROOT / "queue"


def create_queue_files():
    """Create a queue .json file for each track needing choreography."""
    QUEUE_DIR.mkdir(exist_ok=True)
    tracks = get_all_tracks()
    needing = filter_tracks_needing_choreography(tracks)
    print(f"Found {len(needing)} tracks needing choreography.")
    for track in needing:
        spotify_id = track.get("spotify_id")
        if not spotify_id:
            continue
        queue_file = QUEUE_DIR / f"{spotify_id}.json"
        if queue_file.exists():
            continue  # Don't overwrite existing queue files
        with open(queue_file, "w", encoding="utf-8") as f:
            json.dump(track, f, indent=2)
        print(f"  ✓ Queued: {queue_file.name}")


def main():
    create_queue_files()
    process_queue()


if __name__ == "__main__":
    main()
