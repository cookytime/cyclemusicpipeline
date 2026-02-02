#!/bin/bash
librespot --name "CycleMusicLibrespot" &
LIBRESPOT_PID=$!
sleep 2  # Give librespot time to start

python3 capture/auto_capture_playlist_only.py

kill $LIBRESPOT_PID