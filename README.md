# Chroemaker - Spotify Audio Capture & Choreography Generator

Automated audio capture from Spotify with music analysis and choreography generation.

## 🚀 Quick Start (Local)

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

2. Activate the virtual environment:
   - On Linux/Mac:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment:
   ```bash
   cp .env.example .env
   ```

5. Run the main script:
   ```bash
   python3 main.py
   ```

## 📋 Requirements

- Python 3.11+
- FFmpeg with PulseAudio support
- PulseAudio virtual sink
- Spotify desktop app or web browser
- Spotify account
- Spotify Developer credentials
- OpenAI API key

## 🔊 PulseAudio Virtual Sink

```bash
# Create virtual sink
pactl load-module module-null-sink sink_name=spotify_sink sink_properties=device.description="Spotify_Sink"

# Create loopback to hear audio
pactl load-module module-loopback source=spotify_sink.monitor sink=@DEFAULT_SINK@
```

Set Spotify to output to "Spotify_Sink" in your audio settings.

## 📝 Environment Variables

**Required:**
- `SPOTIFY_CLIENT_ID` - From Spotify Developer Dashboard
- `SPOTIFY_CLIENT_SECRET` - From Spotify Developer Dashboard
- `SPOTIFY_REFRESH_TOKEN` - OAuth refresh token
- `SPOTIFY_PLAYLIST_URL` - Playlist URL or ID
- `OPENAI_API_KEY` - For choreography generation

**Optional:**
- `AUTO_UPLOAD=1` - Upload to Base44 after analysis
- `PULSE_MONITOR_SOURCE` - PulseAudio source (default: `spotify_sink.monitor`)
- `POLL_SECONDS` - API polling interval (default: `1.0`)
- `PAD_SECONDS` - Recording padding (default: `1.0`)

## 🎯 Features

- ✅ Automated Spotify playlist capture
- ✅ Audio analysis with librosa
- ✅ AI-generated choreography via OpenAI
- ✅ Headless browser automation
- ✅ Batch processing support
- ✅ Metadata extraction

## ▶️ Common Workflows

### Playlist Capture

```bash
python3 capture/auto_capture_playlist_only.py
```

## 📁 Project Structure

```
.
├── main.py           # Main application entry point
├── requirements.txt  # Python dependencies
├── README.md        # This file
└── .gitignore       # Git ignore rules
```

## 🤝 Contributing

Feel free to open issues or submit pull requests!

## 📄 License

See LICENSE file for details.
