#!/usr/bin/env python3
"""
manage_processing_playlist.py

- Fetches tracks from Base44 that need choreography
- Updates the Spotify processing playlist to match
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from manage.base44_utils import get_all_tracks, filter_tracks_needing_choreography
from manage.spotify_api import refresh_access_token, spotify_get, spotify_put

# Load environment variables
load_dotenv()

from typing import cast

SPOTIFY_PLAYLIST_ID = os.environ.get("SPOTIFY_PLAYLIST_ID") or os.environ.get("SPOTIFY_PLAYLIST_URL")
SPOTIFY_CLIENT_ID = cast(str, os.environ.get("SPOTIFY_CLIENT_ID"))
SPOTIFY_CLIENT_SECRET = cast(str, os.environ.get("SPOTIFY_CLIENT_SECRET"))
SPOTIFY_REFRESH_TOKEN = cast(str, os.environ.get("SPOTIFY_REFRESH_TOKEN"))

if not (SPOTIFY_PLAYLIST_ID and SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET and SPOTIFY_REFRESH_TOKEN):
    print("Missing required Spotify environment variables.")
    sys.exit(1)

def extract_playlist_id(playlist_url: str) -> str:
    import re
    url_match = re.search(r"playlist[:/]([a-zA-Z0-9]+)", playlist_url)
    if url_match:
        return url_match.group(1)
    if re.match(r"^[a-zA-Z0-9]+$", playlist_url):
        return playlist_url
    raise ValueError(f"Invalid Spotify playlist URL or ID: {playlist_url}")

PLAYLIST_ID = extract_playlist_id(SPOTIFY_PLAYLIST_ID)

def main():
    print("Fetching tracks from Base44 needing choreography...")
    all_tracks = get_all_tracks()
    tracks_to_add = filter_tracks_needing_choreography(all_tracks)

    # Only include valid Spotify IDs (22 alphanumeric chars)
    import re
    track_ids = [
        t["spotify_id"]
        for t in tracks_to_add
        if t.get("spotify_id") and re.fullmatch(r"[A-Za-z0-9]{22}", t["spotify_id"])
    ]
    print(f"Tracks to add to playlist: {track_ids}")

    # Get Spotify access token
    token = refresh_access_token(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REFRESH_TOKEN)

    # Build Spotify track URIs
    uris = [f"spotify:track:{tid}" for tid in track_ids]

    # Update the playlist (replace all tracks)
    url = f"https://api.spotify.com/v1/playlists/{PLAYLIST_ID}/tracks"
    print(f"Updating Spotify playlist {PLAYLIST_ID} with {len(uris)} tracks...")
    resp = spotify_put(token, url, data={"uris": uris})
    print("Playlist update response:", resp)

if __name__ == "__main__":
    main()
