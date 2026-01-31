#!/usr/bin/env python3
"""Batch process all WAV files in captures/ to create music maps and POST to n8n webhook."""

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CAPTURES_DIR = Path("captures")
ESSENTIA_SCRIPT = Path("essentia_music_map.py")

# Set this env var or hardcode it:
# export N8N_WEBHOOK_URL="http://localhost:5678/webhook/music-map"
N8N_WEBHOOK_URL = "https://n8n.glencook.tech/webhook-test/music-map"


def get_webhook_url() -> str:
    import os

    url = os.environ.get("N8N_WEBHOOK_URL", N8N_WEBHOOK_URL).strip()
    if not url:
        raise SystemExit(
            "Missing N8N_WEBHOOK_URL env var (e.g. http://localhost:5678/webhook/music-map)"
        )
    return url


def extract_spotify_track_id_from_name(name: str) -> str | None:
    # If your filename includes "... - <spotify_id>.wav" this will grab it.
    # Spotify IDs are base62 ~22 chars.
    m = re.search(r"\b([0-9A-Za-z]{22})\b", name)
    return m.group(1) if m else None


def post_json(url: str, payload: dict, retries: int = 2, debug: bool = True) -> None:
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )

    if debug:
        print(f"  → Webhook URL: {url}")
        print(f"  → Payload size: {len(body)} bytes")
        print(f"  → Track ID: {payload.get('spotify_track_id', 'N/A')}")
        print(f"  → BPM: {payload.get('music_map', {}).get('bpm', 'N/A')}")
        # Show first 500 chars of request body
        body_preview = body.decode("utf-8")[:500]
        print(f"  → Request body preview: {body_preview}...")
        if len(body) > 500:
            print(f"  → (truncated, showing first 500 of {len(body)} bytes)")

    last_err = None
    for attempt in range(retries + 1):
        try:
            if debug and attempt > 0:
                print(f"  → Retry attempt {attempt}/{retries}")

            with urlopen(req, timeout=15) as resp:
                status = resp.status
                response_body = resp.read().decode("utf-8")

                if debug:
                    print(f"  → HTTP {status}")
                    if response_body:
                        print(f"  → Response: {response_body[:200]}")
            return
        except HTTPError as e:
            last_err = e
            if debug:
                error_body = e.read().decode("utf-8") if hasattr(e, "read") else ""
                print(f"  ✗ HTTP Error {e.code}: {e.reason}")
                if error_body:
                    print(f"  ✗ Error body: {error_body[:200]}")
            time.sleep(1.5 * (attempt + 1))
        except URLError as e:
            last_err = e
            if debug:
                print(f"  ✗ URL Error: {e.reason}")
            time.sleep(1.5 * (attempt + 1))
    raise last_err


def main():
    webhook_url = get_webhook_url()

    wav_files = sorted(CAPTURES_DIR.glob("*.wav"))
    if not wav_files:
        print("No WAV files found in captures/")
        return

    to_process = []
    for wav in wav_files:
        json_path = wav.with_suffix(".music_map.json")
        if not json_path.exists():
            to_process.append(wav)

    if not to_process:
        print(f"All {len(wav_files)} WAV files already have music maps!")
        return

    print(f"Found {len(to_process)} files to process (out of {len(wav_files)} total)\n")

    for i, wav in enumerate(to_process, 1):
        json_path = wav.with_suffix(".music_map.json")
        print(f"[{i}/{len(to_process)}] Processing: {wav.name}")

        start = time.time()
        cmd = [sys.executable, str(ESSENTIA_SCRIPT), str(wav), "--out", str(json_path)]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            elapsed = time.time() - start

            if result.returncode == 0 and json_path.exists():
                print(f"  ✓ Map created in {elapsed:.1f}s")

                # Load and POST to n8n
                music_map = json.loads(json_path.read_text(encoding="utf-8"))
                spotify_track_id = extract_spotify_track_id_from_name(wav.name)

                payload = {
                    "type": "music_map",
                    "wav_file": wav.name,
                    "spotify_track_id": spotify_track_id,
                    "music_map": music_map,
                }

                try:
                    print(f"  → Posting to webhook...")
                    post_json(webhook_url, payload)
                    print("  ✓ Posted to n8n\n")
                except Exception as e:
                    print(f"  ✗ Post to n8n failed: {e}\n")
                    import traceback

                    print(f"  ✗ Traceback: {traceback.format_exc()[:300]}\n")

            else:
                print(f"  ✗ Failed after {elapsed:.1f}s")
                if result.stderr:
                    print(f"     Error: {result.stderr[:400]}\n")

        except subprocess.TimeoutExpired:
            print("  ✗ Timeout after 180s\n")
        except Exception as e:
            print(f"  ✗ Error: {e}\n")

    completed = sum(1 for w in wav_files if w.with_suffix(".music_map.json").exists())
    print(f"\nDone! {completed}/{len(wav_files)} music maps created.")


if __name__ == "__main__":
    main()
