"""Shared utilities for Base44 API interactions."""

import os

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_BASE_URL = "https://app.base44.com/api"
APP_ID = os.getenv("BASE44_APP_ID", "69668795c37a96600dabcc5c")
API_KEY = os.getenv("BASE44_API_KEY", "812cc3cbff9a402a9695634f8578398b")
ENTITY_TYPE = "Track"


def make_api_request(api_path, method="GET", data=None, params=None):
    """Make authenticated API request to Base44."""
    url = f"{API_BASE_URL}/{api_path}"
    headers = {"api_key": API_KEY, "Content-Type": "application/json"}

    if method.upper() == "GET":
        response = requests.get(url, headers=headers, params=params, timeout=30)
    else:
        response = requests.request(method, url, headers=headers, json=data, timeout=30)

    response.raise_for_status()
    return response.json()


def get_all_tracks():
    """Fetch all tracks from Base44."""
    try:
        entities = make_api_request(f"apps/{APP_ID}/entities/{ENTITY_TYPE}")
        return entities
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error fetching tracks: {e}")
        return []


def filter_tracks_needing_choreography(tracks):
    """
    Filter tracks that need choreography generation.
    Returns tracks that have:
    - A spotify_id
    - NO choreography array (or empty)
    """
    needing_choreo = []

    for track in tracks:
        spotify_id = track.get("spotify_id")
        choreography = track.get("choreography")

        # Must have Spotify ID
        if not spotify_id:
            continue

        # Must be missing or have empty choreography
        if not choreography or (
            isinstance(choreography, list) and len(choreography) == 0
        ):
            needing_choreo.append(track)

    return needing_choreo


def get_track_spotify_ids_needing_choreography(tracks):
    """
    Filter tracks that need choreography and return only their Spotify IDs.
    Returns list of spotify_id strings.
    """
    spotify_ids = []

    for track in tracks:
        spotify_id = track.get("spotify_id")
        choreography = track.get("choreography")

        # Must have Spotify ID
        if not spotify_id:
            continue

        # Must be missing or have empty choreography
        if not choreography or (
            isinstance(choreography, list) and len(choreography) == 0
        ):
            spotify_ids.append(spotify_id)

    return spotify_ids
