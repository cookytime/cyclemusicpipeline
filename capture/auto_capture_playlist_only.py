#!/usr/bin/env python3
import json
import os
import re
import signal
import subprocess
import sys
import time
import psutil  # You may need to install this: pip install psutil
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# ===== CONFIG =====
# Pulse/PipeWire monitor source for your virtual sink
PULSE_MONITOR_SOURCE = os.environ.get("PULSE_MONITOR_SOURCE", "auto_null.monitor")

# Where to save captures
OUT_DIR = Path(os.environ.get("OUT_DIR", "./captures"))

# Poll interval for Spotify currently-playing
POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "1.0"))

# Record a tiny bit extra so we don't cut off the tail
PAD_SECONDS = float(os.environ.get("PAD_SECONDS", "1.0"))

# Path to your track analyzer script
PROJECT_ROOT = Path(__file__).parent.parent
ANALYZE_SCRIPT = PROJECT_ROOT / "analyze" / "analyze_track.py"
TRACKUPDATE_SCRIPT = PROJECT_ROOT / "manage" / "trackupdate.py"

# Optional: upload results to Base44 after analysis
AUTO_UPLOAD = os.environ.get("AUTO_UPLOAD", "").lower() in {"1", "true", "yes"}

# Required env vars:
#   SPOTIFY_CLIENT_ID
#   SPOTIFY_CLIENT_SECRET
#   SPOTIFY_REFRESH_TOKEN
#   SPOTIFY_PLAYLIST_URL (can be full URL or just ID)
SPOTIFY_CLIENT_ID = os.environ["SPOTIFY_CLIENT_ID"]
SPOTIFY_CLIENT_SECRET = os.environ["SPOTIFY_CLIENT_SECRET"]
SPOTIFY_REFRESH_TOKEN = os.environ["SPOTIFY_REFRESH_TOKEN"]
PLAYLIST_URL = os.environ["SPOTIFY_PLAYLIST_URL"]


def extract_playlist_id(playlist_url: str) -> str:
    """
    Extract playlist ID from Spotify URL or return as-is if already an ID.

    Supports:
    - https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M
    - spotify:playlist:37i9dQZF1DXcBWIGoYBM5M
    - 37i9dQZF1DXcBWIGoYBM5M (direct ID)
    """
    # Try to extract from URL
    url_match = re.search(r"playlist[:/]([a-zA-Z0-9]+)", playlist_url)
    if url_match:
        return url_match.group(1)

    # Assume it's already an ID if it's alphanumeric
    if re.match(r"^[a-zA-Z0-9]+$", playlist_url):
        return playlist_url

    raise ValueError(f"Invalid Spotify playlist URL or ID: {playlist_url}")


PLAYLIST_ID = extract_playlist_id(PLAYLIST_URL)


def extract_spotify_id(filename: str) -> str | None:
    """Extract Spotify ID from filename (22-character base62 string)."""
    match = re.search(r"\b([0-9A-Za-z]{22})\b", filename)
    return match.group(1) if match else None


def refresh_access_token() -> str:
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": SPOTIFY_REFRESH_TOKEN,
            "client_id": SPOTIFY_CLIENT_ID,
            "client_secret": SPOTIFY_CLIENT_SECRET,
        },
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def spotify_get(token: str, url: str, params=None) -> dict:
    r = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=10,
    )
    if r.status_code == 401:
        raise PermissionError("Spotify token expired/unauthorized")
    r.raise_for_status()
    return r.json()


def get_currently_playing(token: str) -> dict | None:
    r = requests.get(
        "https://api.spotify.com/v1/me/player/currently-playing",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    if r.status_code == 204:
        return None
    if r.status_code == 401:
        raise PermissionError("Spotify token expired/unauthorized")
    r.raise_for_status()
    return r.json()


def fetch_playlist_track_ids(token: str, playlist_id: str) -> set[str]:
    """
    Fetch all track IDs from a playlist (pagination-safe).
    """
    ids: set[str] = set()
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    params = {"fields": "items(track(id,type)),next", "limit": 100}

    while True:
        data = spotify_get(token, url, params=params)
        for it in data.get("items", []):
            tr = it.get("track") or {}
            if tr.get("type") == "track" and tr.get("id"):
                ids.add(tr["id"])

        nxt = data.get("next")
        if not nxt:
            break
        url = nxt
        params = None

    return ids


def save_track_metadata(track_item: dict, metadata_path: Path) -> None:
    """
    Save track metadata to JSON file.
    """
    metadata = {
        "spotify_id": track_item.get("id"),
        "name": track_item.get("name"),
        "artists": [
            {"name": a.get("name"), "id": a.get("id"), "uri": a.get("uri")}
            for a in (track_item.get("artists") or [])
        ],
        "album": {
            "name": track_item.get("album", {}).get("name"),
            "id": track_item.get("album", {}).get("id"),
            "release_date": track_item.get("album", {}).get("release_date"),
            "images": track_item.get("album", {}).get("images", []),
        },
        "duration_ms": track_item.get("duration_ms"),
        "explicit": track_item.get("explicit"),
        "uri": track_item.get("uri"),
        "external_urls": track_item.get("external_urls"),
        "isrc": track_item.get("external_ids", {}).get("isrc"),
        "captured_at": datetime.now().isoformat(),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def start_ffmpeg(wav_path: Path, seconds: float) -> subprocess.Popen:
    """
    Record from the monitor source for 'seconds' into wav_path.
    Uses -t so ffmpeg exits automatically.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "pulse",
        "-i",
        PULSE_MONITOR_SOURCE,
        "-ac",
        "1",
        "-ar",
        "44100",
        "-t",
        f"{max(1.0, seconds):.2f}",
        "-y",
        str(wav_path),
    ]
    return subprocess.Popen(cmd)


def run_analyzer(wav_path: Path) -> bool:
    """
    Run the analyze_track.py script which:
    1. Generates music_map.json
    2. Generates choreography.json via OpenAI
    3. Updates Base44 (if configured)

    Returns True if successful, False otherwise.
    """
    print("  Analyzing track and generating choreography...")
    start_time = time.time()

    cmd = [sys.executable, str(ANALYZE_SCRIPT), str(wav_path)]

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Show a spinner while processing
        with tqdm(
            total=100, desc="  Processing", bar_format="{desc}: {elapsed}", leave=False
        ) as pbar:
            while proc.poll() is None:
                time.sleep(0.1)
                pbar.update(0)

        stdout, stderr = proc.communicate()

        if proc.returncode != 0:
            print(f"  ❌ Analysis failed:")
            if stdout:
                print(f"     {stdout[:200]}")
            if stderr:
                print(f"     {stderr[:200]}")
            return False

        elapsed = time.time() - start_time
        print(f"  ✓ Analysis complete in {elapsed:.1f}s")

        # Show key outputs
        base_path = wav_path.with_suffix("")
        if (base_path.with_suffix(".music_map.json")).exists():
            print(f"  ✓ Music map saved")
        if (base_path.with_suffix(".choreography.json")).exists():
            print(f"  ✓ Choreography saved")

        if AUTO_UPLOAD:
            spotify_id = extract_spotify_id(wav_path.name)
            if spotify_id:
                print(f"  Uploading to Base44 (spotify_id={spotify_id})...")
                upload_cmd = [
                    sys.executable,
                    str(TRACKUPDATE_SCRIPT),
                    spotify_id,
                    str(wav_path.parent),
                ]
                try:
                    upload_result = subprocess.run(
                        upload_cmd, check=True, capture_output=True, text=True
                    )
                    if upload_result.stdout:
                        print(upload_result.stdout)
                    if upload_result.stderr:
                        print(f"  Upload warnings: {upload_result.stderr}")
                except subprocess.CalledProcessError as e:
                    print(f"  ❌ Upload failed: {e}")
                    if e.stdout:
                        print(e.stdout)
                    if e.stderr:
                        print(e.stderr)
            else:
                print("  ⚠️  No Spotify ID found; skipping Base44 upload")

        return True

    except Exception as e:
        print(f"  ❌ Analysis error: {e}")
        return False


def stop_proc(proc: subprocess.Popen | None) -> None:
    if not proc:
        return
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def is_librpspot_running():
    """Check if librpspot is already running."""
    for proc in psutil.process_iter(['name', 'exe', 'cmdline']):
        try:
            if 'librpspot' in proc.info['name'] or \
               (proc.info['exe'] and 'librpspot' in proc.info['exe']) or \
               (proc.info['cmdline'] and any('librpspot' in str(arg) for arg in proc.info['cmdline'])):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False

def start_librpspot():
    """Start librpspot as a subprocess if not already running."""
    librpspot_path = "/usr/local/bin/librpspot"
    print(f"Trying to start librpspot at: {librpspot_path}")
    if not Path(librpspot_path).exists():
        print(f"✗ librpspot not found at {librpspot_path}")
        return None
    if is_librpspot_running():
        print("✓ librpspot is already running.")
        return None
    try:
        proc = subprocess.Popen([librpspot_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✓ librpspot started.")
        time.sleep(3)
        if proc.poll() is not None:
            out, err = proc.communicate()
            print("✗ librpspot exited immediately.")
            print("stdout:", out.decode())
            print("stderr:", err.decode())
            return None
        return proc
    except Exception as e:
        print(f"✗ Failed to start librpspot: {e}")
        return None

def stop_librpspot(proc):
    """Stop the librpspot subprocess if we started it."""
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
            print("✓ librpspot stopped.")
        except subprocess.TimeoutExpired:
            proc.kill()
            print("✓ librpspot killed.")

def main():
    print("DEBUG: main() started")
    librpspot_proc = start_librpspot()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify analyzer script exists
    if not ANALYZE_SCRIPT.exists():
        print(f"❌ Analyzer script not found: {ANALYZE_SCRIPT}")
        print("   Expected location: analyze/analyze_track.py")
        sys.exit(1)

    token = refresh_access_token()
    playlist_track_ids = fetch_playlist_track_ids(token, PLAYLIST_ID)
    playlist_uri = f"spotify:playlist:{PLAYLIST_ID}"

    print(f"Loaded {len(playlist_track_ids)} tracks from playlist: {PLAYLIST_ID}")
    print(f"Capturing audio from: {PULSE_MONITOR_SOURCE}")
    print(f"Writing files to: {OUT_DIR.resolve()}")
    print(f"Analyzer: {ANALYZE_SCRIPT.relative_to(PROJECT_ROOT)}")
    print("Tip: start playback by pressing Play from the playlist header.\n")

    current_track_id: str | None = None
    ffmpeg_proc: subprocess.Popen | None = None
    current_wav: Path | None = None
    current_item: dict | None = None

    try:
        while True:
            # Get currently playing
            try:
                payload = get_currently_playing(token)
            except PermissionError:
                token = refresh_access_token()
                payload = get_currently_playing(token)

            if not payload or not payload.get("is_playing"):
                # If nothing playing, ensure we aren't recording
                stop_proc(ffmpeg_proc)
                ffmpeg_proc = None
                current_wav = None
                current_track_id = None
                current_item = None
                time.sleep(POLL_SECONDS)
                continue

            item = payload.get("item")
            ctx = payload.get("context") or {}
            ctx_uri = ctx.get("uri")

            # We only handle track items
            if not item or item.get("type") != "track":
                time.sleep(POLL_SECONDS)
                continue

            track_id = item.get("id")
            if not track_id:
                time.sleep(POLL_SECONDS)
                continue

            # Gate 1: must be playing FROM the playlist context
            if ctx_uri != playlist_uri:
                # Not in our playlist context → stop any capture
                if ffmpeg_proc and ffmpeg_proc.poll() is None:
                    print("⏸ Not in playlist context; stopping capture.")
                stop_proc(ffmpeg_proc)
                ffmpeg_proc = None
                current_wav = None
                current_track_id = None
                current_item = None
                time.sleep(POLL_SECONDS)
                continue

            # Gate 2: track must belong to that playlist
            if track_id not in playlist_track_ids:
                time.sleep(POLL_SECONDS)
                continue

            # Track timing
            progress_ms = payload.get("progress_ms") or 0
            duration_ms = item.get("duration_ms") or 0
            remaining_s = max(0.0, (duration_ms - progress_ms) / 1000.0) + PAD_SECONDS

            name = item.get("name", "Unknown Track")
            artists = (
                ", ".join(
                    a.get("name", "")
                    for a in (item.get("artists") or [])
                    if a.get("name")
                )
                or "Unknown Artist"
            )

            # If track changed → start new capture
            if track_id != current_track_id:
                # Stop any previous capture
                stop_proc(ffmpeg_proc)

                # Use Spotify ID as filename
                current_wav = OUT_DIR / f"{track_id}.wav"
                metadata_path = OUT_DIR / f"{track_id}.metadata.json"

                print(f"▶ Capturing (playlist-only): {artists} — {name}")
                print(f"  Remaining ~{remaining_s:.1f}s → {current_wav.name}")

                # Save metadata
                save_track_metadata(item, metadata_path)
                print(f"  ✓ Metadata saved: {metadata_path.name}")

                ffmpeg_proc = start_ffmpeg(current_wav, remaining_s)
                current_track_id = track_id
                current_item = item

            # If ffmpeg finished, analyze and reset
            if ffmpeg_proc and ffmpeg_proc.poll() is not None and current_wav:
                if current_wav.exists() and current_wav.stat().st_size > 100_000:
                    success = run_analyzer(current_wav)
                    if success:
                        print(f"✅ Track fully processed\n")
                    else:
                        print(f"⚠️ Track captured but analysis failed\n")
                else:
                    print(f"⚠️ Skipped tiny capture: {current_wav.name}\n")

                ffmpeg_proc = None
                current_wav = None
                current_track_id = None
                current_item = None

            time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        print("\nStopping…")
        stop_proc(ffmpeg_proc)


if __name__ == "__main__":
    main()
