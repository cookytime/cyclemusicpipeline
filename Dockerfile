FROM ubuntu:22.04

# Install Python 3.11, pip, Java, and system dependencies
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        ffmpeg \
        pulseaudio \
        pulseaudio-utils \
        libsndfile1 \
        ca-certificates \
        openjdk-17-jre \
        gnupg2 \
        wget \
        curl \
        lsb-release \
        build-essential \
        python3.11 \
        python3.11-venv \
        python3.11-distutils \
        python3-pip \
        python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN python3 --version && pip3 --version
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps:
# - ffmpeg: capture
# - pulseaudio + pactl: null sink + monitor
# - librespot: Spotify Connect playback inside container
# - libsndfile1: librosa/soundfile backend
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      pulseaudio \
      pulseaudio-utils \
      libsndfile1 \
      ca-certificates \
      openjdk-17-jre \
      gnupg2 \
    && rm -rf /var/lib/apt/lists/*


# Add spocon PPA and install spocon
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 7DBE8BF06EA39B78 && \
    echo 'deb http://ppa.launchpad.net/spocon/spocon/ubuntu bionic main' | tee /etc/apt/sources.list.d/spocon.list && \
    apt-get update && \
    apt-get install -y spocon && \
    rm -rf /var/lib/apt/lists/*


WORKDIR /app

# Install python deps
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Runtime dirs
RUN mkdir -p /app/captures /tmp/runtime

# Replace entrypoint with headless Connect+Pulse runner
COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Defaults (override via env)
ENV XDG_RUNTIME_DIR=/tmp/runtime \
    OUT_DIR=/app/captures \
    PULSE_MONITOR_SOURCE=spotify_sink.monitor \
    SPOTIFY_CONNECT_NAME=TrueNAS-Analyzer \
    AUTO_UPLOAD=1 \
    AUTO_START_PLAYBACK=1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python3", "capture/auto_capture_playlist_only.py"]
