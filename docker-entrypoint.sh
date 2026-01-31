#!/usr/bin/env bash
set -euo pipefail

echo "🎵 Starting Chroemaker (headless Spotify Connect + PulseAudio)..."

# ----- Runtime dirs for PulseAudio -----
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime}"
mkdir -p "$XDG_RUNTIME_DIR/pulse"
chmod 700 "$XDG_RUNTIME_DIR"

# Make sure Pulse uses a predictable socket path
export PULSE_SERVER="unix:${XDG_RUNTIME_DIR}/pulse/native"

# ----- Start PulseAudio (user mode) -----
# On servers/containers, user-mode PA is more predictable than --system.
# We create a null sink for capture and expose a unix socket for local clients.
cat > /tmp/pulseaudio.pa <<'EOF'
load-module module-native-protocol-unix auth-anonymous=1 socket=/tmp/runtime/pulse/native
load-module module-null-sink sink_name=spotify_sink sink_properties=device.description=Spotify_Sink
load-module module-always-sink
EOF

pulseaudio -D --exit-idle-time=-1 --log-level=error --load=/tmp/pulseaudio.pa

# Wait for PA socket
for i in $(seq 1 40); do
  if pactl info >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done

echo "✅ PulseAudio ready. Sinks:"
pactl list short sinks || true

# ----- Start Spotify Connect (spocon) into spotify_sink -----
CONNECT_NAME="${SPOTIFY_CONNECT_NAME:-TrueNAS-Analyzer}"
BITRATE="${SPOCON_BITRATE:-160}"

if [[ -z "${SPOCON_USERNAME:-}" || -z "${SPOCON_PASSWORD:-}" ]]; then
  echo "❌ Missing SPOCON_USERNAME or SPOCON_PASSWORD"
  exit 2
fi

echo "🔊 Starting spocon device: ${CONNECT_NAME}"
/opt/spocon/spocon \
  --name "${CONNECT_NAME}" \
  --backend pulseaudio \
  --device spotify_sink \
  --bitrate "${BITRATE}" \
  --username "${SPOCON_USERNAME}" \
  --password "${SPOCON_PASSWORD}" \
  >/tmp/spocon.log 2>&1 &
LOG_DIR="/app/captures/spocon-logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/spocon_$(date +%Y%m%d_%H%M%S).log"
ERR_FILE="$LOG_DIR/spocon_error_$(date +%Y%m%d_%H%M%S).log"
echo "[DEBUG] Spocon log: $LOG_FILE"
echo "[DEBUG] Spocon error log: $ERR_FILE"
env | grep SPOCON || true
which spocon || echo "[DEBUG] spocon not found in PATH"
spocon \
  --name "${CONNECT_NAME}" \
  --backend pulseaudio \
  --device spotify_sink \
  --bitrate "${BITRATE}" \
  --username "${SPOCON_USERNAME}" \
  --password "${SPOCON_PASSWORD}" \
  > >(tee "$LOG_FILE") 2> >(tee "$ERR_FILE" >&2) &
SPOCON_PID=$!

# ----- Optional: transfer playback + start playlist on this Connect device -----
AUTO_START="${AUTO_START_PLAYBACK:-1}"
if [[ "${AUTO_START,,}" == "true" || "${AUTO_START}" == "1" ]]; then
  if [[ -n "${SPOTIFY_CLIENT_ID:-}" && -n "${SPOTIFY_CLIENT_SECRET:-}" && -n "${SPOTIFY_REFRESH_TOKEN:-}" && -n "${SPOTIFY_PLAYLIST_URL:-}" ]]; then
    echo "▶️  AUTO_START_PLAYBACK enabled - starting playlist on '${CONNECT_NAME}'..."
    python - <<'PY'
import os, re, time, requests, sys

cid = os.environ["SPOTIFY_CLIENT_ID"]
sec = os.environ["SPOTIFY_CLIENT_SECRET"]
rt  = os.environ["SPOTIFY_REFRESH_TOKEN"]
playlist = os.environ["SPOTIFY_PLAYLIST_URL"]
device_name = os.environ.get("SPOTIFY_CONNECT_NAME", "TrueNAS-Analyzer")

def extract_playlist_id(s: str) -> str:
    m = re.search(r"playlist[:/]([A-Za-z0-9]+)", s)
    if m: return m.group(1)
    if re.fullmatch(r"[A-Za-z0-9]+", s): return s
    raise ValueError(f"Bad SPOTIFY_PLAYLIST_URL: {s}")

plid = extract_playlist_id(playlist)

def refresh_token():
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": rt,
            "client_id": cid,
            "client_secret": sec,
        },
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["access_token"]

token = refresh_token()
hdr = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

device_id = None
devices = []
for _ in range(15):
    r = requests.get("https://api.spotify.com/v1/me/player/devices", headers=hdr, timeout=15)
    r.raise_for_status()
    devices = r.json().get("devices", [])
    for d in devices:
        if d.get("name") == device_name:
            device_id = d.get("id")
            break
    if device_id:
        break
    time.sleep(1)

if not device_id:
    print(f"[auto-start] Could not find device named '{device_name}'. Devices: {[d.get('name') for d in devices]}", file=sys.stderr)
    sys.exit(2)

# Transfer playback (do not start yet)
requests.put(
    "https://api.spotify.com/v1/me/player",
    headers=hdr,
    json={"device_ids": [device_id], "play": False},
    timeout=15,
).raise_for_status()

# Start playlist
requests.put(
    "https://api.spotify.com/v1/me/player/play",
    headers=hdr,
    json={"context_uri": f"spotify:playlist:{plid}"},
    timeout=15,
).raise_for_status()

print(f"[auto-start] Playback started on '{device_name}' for playlist {plid}")
PY
  else
    echo "⚠️  AUTO_START_PLAYBACK enabled but Spotify Web API env vars missing; skipping auto-start."
  fi
fi

# ----- Run pipeline (capture → analyze → GPT → Base44) -----
# capture/auto_capture_playlist_only.py:
#  - captures from PULSE_MONITOR_SOURCE using ffmpeg
#  - runs analyze_track.py (music map + GPT choreography)
#  - if AUTO_UPLOAD=1, calls manage/trackupdate.py to push to Base44
echo "🚀 Running: $*"
"$@" || EXIT_CODE=$?


# Cleanup
echo "🧹 Shutting down spocon..."
kill "${SPOCON_PID}" >/dev/null 2>&1 || true

exit "${EXIT_CODE:-0}"
