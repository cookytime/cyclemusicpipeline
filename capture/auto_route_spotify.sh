#!/usr/bin/env bash

# Check if sink exists, create if needed
if ! pactl list sinks short | grep -q spotify_sink; then
    echo "Creating spotify_sink..."
    pactl load-module module-null-sink sink_name=spotify_sink sink_properties=device.description="Spotify_Sink"
fi

echo "Monitoring for Spotify audio streams..."
echo "Start playing something in Spotify to route audio."

while true; do
    ID=$(pactl list sink-inputs | awk '
        /Sink Input #/ {id=$3; gsub(/#/, "", id)}
        /application.process.binary = "spotify"/ {print id}
    ')

    if [ -n "$ID" ]; then
        echo "$(date '+%H:%M:%S') - Found Spotify stream #$ID, routing to spotify_sink"
        pactl move-sink-input "$ID" spotify_sink 2>/dev/null
    fi

    sleep 1
done
