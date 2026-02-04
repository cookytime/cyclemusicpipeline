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


## ▶️ Full Pipeline Workflow

This project is designed to automate the process of syncing a Spotify playlist, capturing audio, analyzing tracks, generating choreography, and uploading results to Base44. The recommended workflow is:

1. **Sync the processing playlist with Base44**
   ```bash
   python3 main.py sync-playlist
   ```
   This updates the playlist of tracks needing choreography in Base44.

2. **Start librespot and prepare Spotify**
   - Launch librespot (or your preferred Spotify client) and ensure it is ready to play the synced playlist.
   - Wait for user input to press play on the playlist.

3. **Capture the playlist audio**
   ```bash
   python3 main.py capture-playlist
   ```
   This will record all tracks in the playlist to the captures/ directory.

4. **Analyze and upload tracks**
   ```bash
   python3 main.py process-captures
   ```
   This step analyzes each WAV file, generates music map and choreography, and uploads results to Base44.

You can also run `python3 main.py` with no arguments to process all captured tracks by default.

### Example: Full Orchestration

```bash
# 1. Sync playlist
python3 main.py sync-playlist

# 2. Start librespot and press play on the playlist

# 3. Capture playlist
python3 main.py capture-playlist

# 4. Analyze and upload
python3 main.py process-captures
```

## 🐳 Docker Usage

This project includes a Dockerfile and docker-compose.yml for containerized execution. To build and run:

```bash
docker-compose build
docker-compose up
```

Ensure your .env file is configured with all required environment variables before running Docker.


## 📁 Project Structure

```
.
├── main.py                 # Main orchestration entry point
├── analyze/                # Audio analysis and choreography generation
├── capture/                # Audio capture scripts
├── manage/                 # Base44 and playlist management
├── captures/               # Captured audio and analysis results
├── requirements.txt        # Python dependencies
├── Dockerfile, docker-compose.yml  # Containerization
├── README.md               # Project documentation
└── .gitignore              # Git ignore rules
```

## 🤝 Contributing

Feel free to open issues or submit pull requests!

## 📄 License

See LICENSE file for details.
