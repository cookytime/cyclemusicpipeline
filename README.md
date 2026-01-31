# Chroemaker - Spotify Audio Capture & Choreography Generator

Automated audio capture from Spotify with music analysis and choreography generation.

## 🚀 Quick Start (Docker - Recommended)

The easiest way to run this project is with Docker, which includes everything you need:

```bash
# 1. Set up environment
cp .env.example .env
# Edit .env with your credentials

# 2. Build and run
./docker-run.sh

# Or test browser automation first:
./docker-run.sh python spotify_browser.py
```

See [DOCKER_CHROME.md](DOCKER_CHROME.md) for detailed Docker setup with containerized Chrome and PulseAudio.

## 📋 Requirements

### Docker Setup (Recommended)
- Docker and Docker Compose
- Spotify account
- Spotify Developer credentials
- OpenAI API key

### Local Setup (Advanced)
- Python 3.11+
- FFmpeg with PulseAudio support
- PulseAudio virtual sink
- Spotify desktop app or web browser
- Virtual environment

## 🐳 Docker Setup

### 1. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:
- `SPOTIFY_USERNAME` - Your Spotify login
- `SPOTIFY_PASSWORD` - Your Spotify password
- `SPOTIFY_CLIENT_ID` - From [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
- `SPOTIFY_CLIENT_SECRET` - From Spotify Developer Dashboard
- `SPOTIFY_REFRESH_TOKEN` - OAuth refresh token
- `SPOTIFY_PLAYLIST_URL` - Playlist to capture
- `OPENAI_API_KEY` - For choreography generation

### 2. Build Container

```bash
docker-compose build
```

### 3. Run Application

```bash
# Test browser automation
./docker-run.sh python spotify_browser.py

# Run capture workflow
./docker-run.sh python capture/auto_capture_playlist_only.py

# Interactive shell
./docker-run.sh bash
```

### 4. View Browser (Debugging)

Connect with VNC viewer:
```bash
vncviewer localhost:5900
# Password: chroemaker (or your VNC_PASSWORD)
```

## 💻 Local Setup

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

### 2. Configure PulseAudio Virtual Sink

```bash
# Create virtual sink
pactl load-module module-null-sink sink_name=spotify_sink sink_properties=device.description="Spotify_Sink"

# Create loopback to hear audio
pactl load-module module-loopback source=spotify_sink.monitor sink=@DEFAULT_SINK@
```

Set Spotify to output to "Spotify_Sink" in your audio settings.

### 3. Set Environment Variables

Create `.env` file with your credentials (see `.env.example`).

## 📁 Project Structure
## 📝 Environment Variables

**Required:**
- `SPOTIFY_CLIENT_ID` - From Spotify Developer Dashboard
- `SPOTIFY_CLIENT_SECRET` - From Spotify Developer Dashboard
- `SPOTIFY_REFRESH_TOKEN` - OAuth refresh token
- `SPOTIFY_PLAYLIST_URL` - Playlist URL or ID
- `OPENAI_API_KEY` - For choreography generation

**Docker-specific:**
- `SPOTIFY_USERNAME` - Your Spotify email/username
- `SPOTIFY_PASSWORD` - Your Spotify password

**Optional:**
- `AUTO_UPLOAD=1` - Upload to Base44 after analysis
- `PULSE_MONITOR_SOURCE` - PulseAudio source (default: `spotify_sink.monitor`)
- `VNC_PASSWORD` - VNC server password (default: `chroemaker`)
- `POLL_SECONDS` - API polling interval (default: `1.0`)
- `PAD_SECONDS` - Recording padding (default: `1.0`)

## 🎯 Features

- ✅ Automated Spotify playlist capture
- ✅ Audio analysis with librosa
- ✅ AI-generated choreography via OpenAI
- ✅ Fully containerized with Docker
- ✅ Headless browser automation
- ✅ VNC remote viewing
- ✅ Batch processing support
- ✅ Metadata extraction

## 🤝 Contributing

Feel free to open issues or submit pull requests!

## 📄 License

See LICENSE file for details.

## 🙏 Acknowledgments

- Built with Python, Docker, Selenium, FFmpeg, and PulseAudio
- Uses Spotify Web API and OpenAI API requirements.txt     # Python dependencies
├── Dockerfile           # Docker image definition
├── docker-compose.yml   # Docker orchestration
├── docker-entrypoint.sh # Container startup script
├── .env.example         # Environment template
└── README.md            # This file
```

## 🔧 How It Works

### Docker Architecture (Containerized Chrome)

1. **Chrome** runs Spotify Web Player in headless mode
2. **PulseAudio** captures audio from Chrome via virtual sink
3. **FFmpeg** records audio stream to WAV files
4. **Python** analyzes audio and generates choreography
5. **VNC** (optional) allows viewing the browser

### Local Architecture (Host PulseAudio)

1. **Spotify** plays on host machine
2. **PulseAudio** routes audio through virtual sink
3. **FFmpeg** captures from monitor source
4. **Python** analyzes and processes

## 📖 Documentation

- [DOCKER_CHROME.md](DOCKER_CHROME.md) - Detailed Docker setup with Chrome
- [.env.example](.env.example) - Environment variable reference

## 🛠️ Troubleshooting

### Docker Issues

```bash
# Check PulseAudio
docker-compose run --rm chroemaker pactl info

# Check Chrome
docker-compose run --rm chroemaker chromium --version

# View logs
docker-compose logs -f

# Test audio capture
docker-compose run --rm chroemaker ffmpeg -f pulse -list_sources true -i default
```

### Local Issues

- Ensure PulseAudio virtual sink is created
- Verify Spotify is outputting to the sink
- Check FFmpeg can access PulseAudio: `ffmpeg -f pulse -list_sources true -i default`

## 📝 Environment Variables

Execute the main script:
```bash
python3 main.py
```

### Playlist Capture

```bash
python3 capture/auto_capture_playlist_onlymain script:
```bash
python3 main.py
```

### Playlist ripper GUI

Launch a simple GUI to select a Spotify playlist and start a capture session:
```bash
python3 gui_playlist_ripper.py
```

Environment variables required:
- `SPOTIFY_CLIENT_ID`
- `SPOTIFY_CLIENT_SECRET`
- `SPOTIFY_REFRESH_TOKEN`

Optional:
- `AUTO_UPLOAD=1` to upload to Base44 after analysis

## Project Structure

```
.
├── main.py           # Main application entry point
├── requirements.txt  # Python dependencies
├── README.md        # This file
└── .gitignore       # Git ignore rules
```
