#!/usr/bin/env bash
set -euo pipefail

echo "🎵 Starting Chroemaker (headless Spotify Connect + PulseAudio)..."

# ----- Runtime dirs for PulseAudio -----
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime}"
mkdir -p "$XDG_RUNTIME_DIR/pulse"
chmod 700 "$XDG_RUNTIME_DIR"

# Make sure Pulse uses a predictable socket path
export PULSE_SERVER="unix:${XDG_RUNTIME_DIR}/pulse/native"
export PULSE_MONITOR_SOURCE="${PULSE_MONITOR_SOURCE:-auto_null.monitor}"

# ----- Start PulseAudio (user mode) -----
# On servers/containers, user-mode PA is more predictable than --system.
# We create a null sink for capture and expose a unix socket for local clients.
cat > /tmp/pulseaudio.pa <<'EOF'
load-module module-native-protocol-unix auth-anonymous=1 socket=/tmp/runtime/pulse/native
load-module module-null-sink sink_name=auto_null sink_properties=device.description=Music_Sink
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


# ----- Start Spotify Connect (spotifyd) into auto_null -----
echo "🔊 Starting spotifyd device using /etc/spotifyd.conf..."
spotifyd --no-daemon --use-mpris=false --config-path /etc/spotifyd.conf &
SPOTIFYD_PID=$!


# ----- Run pipeline (capture → analyze → GPT → Base44) -----
# capture/auto_capture_playlist_only.py:
#  - captures from PULSE_MONITOR_SOURCE using ffmpeg
#  - runs analyze_track.py (music map + GPT choreography)
#  - if AUTO_UPLOAD=1, calls manage/trackupdate.py to push to Base44

echo "[DEBUG] Entering: Main pipeline command ($*)"
"$@" || EXIT_CODE=$?

echo "[DEBUG] Entering: manage_processing_playlist.py"
python3 manage/manage_processing_playlist.py



# Cleanup
echo "🧹 Shutting down spotifyd..."
kill "${SPOTIFYD_PID}" >/dev/null 2>&1 || true

exit "${EXIT_CODE:-0}"
