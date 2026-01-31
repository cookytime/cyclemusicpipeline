#!/bin/bash
cd /home/glen/Documents/bpmanalizer

total=$(ls captures/*.wav 2>/dev/null | wc -l)
processed=0

echo "Processing $total WAV files..."
echo

for wav in captures/*.wav; do
    json="${wav%.wav}.music_map.json"

    if [ -f "$json" ]; then
        echo "⏭  Skipping $(basename "$wav") - already processed"
        ((processed++))
        continue
    fi

    ((processed++))
    echo "[$processed/$total] Processing: $(basename "$wav")"
    echo "  Started: $(date +%H:%M:%S)"

    start=$(date +%s)
    timeout 120 python essentia_music_map.py "$wav" --out "$json" 2>&1 | grep -v "tensorflow\|cudart\|libcuda\|INFO"
    end=$(date +%s)
    elapsed=$((end - start))

    if [ -f "$json" ]; then
        echo "  ✓ Complete in ${elapsed}s"
    else
        echo "  ✗ Failed or timed out after ${elapsed}s"
    fi
    echo
done

completed=$(ls captures/*.music_map.json 2>/dev/null | wc -l)
echo "Done! $completed/$total music maps created."
