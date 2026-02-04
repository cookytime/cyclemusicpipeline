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

import json
import os
import re
import signal
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import psutil  # You may need to install this: pip install psutil
from dotenv import load_dotenv
from tqdm import tqdm

# Ensure project root is on sys.path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manage.spotify_api import (refresh_access_token, spotify_get,
                                spotify_get_currently_playing)

# Load environment variables from .env file at project root
load_dotenv(PROJECT_ROOT / ".env")


# ===== CONFIG =====
# Pulse/PipeWire monitor source for your virtual sink
PULSE_MONITOR_SOURCE = os.environ.get("PULSE_MONITOR_SOURCE", "librespot_sink.monitor")

# Librespot config
LIBRESPOT_PATH = os.environ.get("LIBRESPOT_PATH", "/usr/bin/librespot")
LIBRESPOT_NAME = os.environ.get("LIBRESPOT_NAME", "CycleMusicLibrespot")
LIBRESPOT_SINK = os.environ.get("LIBRESPOT_SINK", "auto_null")


# Where to save captures
OUT_DIR = Path(os.environ.get("OUT_DIR", "./captures"))
TMP_DIR = Path(os.environ.get("TMP_DIR", "./tmp"))
TMP_DIR.mkdir(exist_ok=True)

# Poll interval for Spotify currently-playing
POLL_SECONDS = float(os.environ.get("POLL_SECONDS", "1.0"))

# Record a tiny bit extra so we don't cut off the tail
PAD_SECONDS = float(os.environ.get("PAD_SECONDS", "1.0"))

# Path to your track analyzer script
ANALYZE_SCRIPT = PROJECT_ROOT / "analyze" / "analyze_track.py"
TRACKUPDATE_SCRIPT = PROJECT_ROOT / "manage" / "trackupdate.py"

# Whether to auto-upload to Base44 after analysis (set via env or default to False)
AUTO_UPLOAD = os.environ.get("AUTO_UPLOAD", "0").lower() in ("1", "true", "yes")
# Run analysis in the background to keep capturing new tracks
ANALYZE_IN_BACKGROUND = os.environ.get("ANALYZE_IN_BACKGROUND", "1").lower() in (
    "1",
    "true",
    "yes",
)


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


def extract_spotify_id(filename: str) -> str | None:
    """Extract Spotify ID from filename (22-character base62 string)."""
    match = re.search(r"\b([0-9A-Za-z]{22})\b", filename)
    return match.group(1) if match else None


def load_spotify_config() -> dict:
    """Load required Spotify config values from the environment."""
    required_vars = [
        "SPOTIFY_CLIENT_ID",
        "SPOTIFY_CLIENT_SECRET",
        "SPOTIFY_REFRESH_TOKEN",
        "SPOTIFY_PLAYLIST_URL",
    ]
    missing = [name for name in required_vars if not os.environ.get(name)]
    if missing:
        raise KeyError(f"Missing Spotify config: {', '.join(missing)}")

    return {
        "client_id": os.environ["SPOTIFY_CLIENT_ID"],
        "client_secret": os.environ["SPOTIFY_CLIENT_SECRET"],
        "refresh_token": os.environ["SPOTIFY_REFRESH_TOKEN"],
        "playlist_url": os.environ["SPOTIFY_PLAYLIST_URL"],
    }


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


def submit_analysis(executor: ThreadPoolExecutor, wav_path: Path) -> Future | None:
    if not ANALYZE_IN_BACKGROUND:
        return None
    print(f"  🧵 Starting background pipeline for {wav_path.name}")
    return executor.submit(run_analyzer, wav_path)


def stop_proc(proc: subprocess.Popen | None) -> None:
    if not proc:
        return
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def is_librespot_running(verbose=True):
    for proc in psutil.process_iter(["pid", "name", "exe", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if any(
                "librespot" in str(x)
                for x in ([proc.info.get("name"), proc.info.get("exe")] + cmdline)
            ):
                if verbose:
                    print(
                        f"Found librespot PID {proc.info['pid']}: {' '.join(map(str, cmdline))}"
                    )
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def start_librespot():
    """Start librespot as a subprocess if not already running."""
    if not Path(LIBRESPOT_PATH).exists():
        print(f"✗ librespot not found at {LIBRESPOT_PATH}")
        return None
    if is_librespot_running():
        print("✓ librespot is already running.")
        return None
    env = os.environ.copy()
    env["PULSE_SINK"] = LIBRESPOT_SINK
    cmd = [
        LIBRESPOT_PATH,
        "--name",
        LIBRESPOT_NAME,
        # Do NOT add --username or --password
    ]

    print(f"Starting librespot: {' '.join(cmd)} with PULSE_SINK={LIBRESPOT_SINK}")
    try:
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        time.sleep(3)
        if proc.poll() is not None:
            out, err = proc.communicate()
            print("✗ librespot exited immediately.")
            print("stdout:", out.decode())
            print("stderr:", err.decode())
            return None
        print("✓ librespot started.")
        return proc
    except Exception as e:
        print(f"✗ Failed to start librespot: {e}")
        return None


def stop_librespot(proc):
    """Stop the librespot subprocess if we started it."""
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
            print("✓ librespot stopped.")
        except subprocess.TimeoutExpired:
            proc.kill()
            print("✓ librespot killed.")


def main():
    print("DEBUG: main() started")
    librespot_proc = start_librespot()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis_executor = ThreadPoolExecutor(max_workers=1)
    analysis_futures: list[Future] = []

    # Verify analyzer script exists
    if not ANALYZE_SCRIPT.exists():
        print(f"❌ Analyzer script not found: {ANALYZE_SCRIPT}")
        print("   Expected location: analyze/analyze_track.py")
        sys.exit(1)

    spotify_config = load_spotify_config()
    playlist_id = extract_playlist_id(spotify_config["playlist_url"])
    token = refresh_access_token(
        spotify_config["client_id"],
        spotify_config["client_secret"],
        spotify_config["refresh_token"],
    )
    playlist_track_ids = fetch_playlist_track_ids(token, playlist_id)
    playlist_uri = f"spotify:playlist:{playlist_id}"

    print(f"Loaded {len(playlist_track_ids)} tracks from playlist: {playlist_id}")
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
                payload = spotify_get_currently_playing(token)
            except PermissionError:
                token = refresh_access_token(
                    spotify_config["client_id"],
                    spotify_config["client_secret"],
                    spotify_config["refresh_token"],
                )
                payload = spotify_get_currently_playing(token)

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

                # Store temp wav in TMP_DIR
                current_wav = TMP_DIR / f"{track_id}.wav"
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
                    # Move temp wav to OUT_DIR after processing
                    final_wav = OUT_DIR / current_wav.name
                    if ANALYZE_IN_BACKGROUND:
                        future = submit_analysis(analysis_executor, current_wav)
                        if future:
                            analysis_futures.append(future)
                    else:
                        success = run_analyzer(current_wav)
                        if success:
                            print(f"✅ Track fully processed\n")
                        else:
                            print(f"⚠️ Track captured but analysis failed\n")
                    # Move file only if analysis succeeded or always?
                    try:
                        import shutil

                        shutil.move(str(current_wav), str(final_wav))
                        print(f"  ✓ Moved {current_wav.name} to captures/")
                    except Exception as e:
                        print(f"⚠️ Failed to move wav file: {e}")
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
        stop_librespot(librespot_proc)
    finally:
        if analysis_futures:
            print("Waiting for background analysis tasks to complete...")
        for future in analysis_futures:
            future.result()
        analysis_executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
