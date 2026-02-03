#!/usr/bin/env python3
import os
import sys
from pathlib import Path


def _reexec_with_venv() -> None:
    if __name__ != "__main__":
        return
    if os.environ.get("VIRTUAL_ENV"):
        return
    start = Path(__file__).resolve().parent
    venv_python = None
    for parent in (start, *start.parents):
        candidate = parent / ".venv" / "bin" / "python"
        if candidate.exists():
            venv_python = candidate
            break
    if venv_python is None:
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_reexec_with_venv()

"""
Manage a special Spotify playlist for tracks needing choreography.
- Creates/updates a "Choreography Queue" playlist
- Adds tracks from Base44 that need choreography
- Removes tracks once they're fully processed
- Limits playlist to N tracks at a time for manageable processing
"""

import os
import sys
from pathlib import Path

from manage.base44_utils import get_all_tracks, get_track_spotify_ids_needing_choreography
from dotenv import load_dotenv
from manage.spotify_api import (
    refresh_access_token,
    spotify_delete,
    spotify_get,
    spotify_post,
    spotify_put,
)

# Load environment variables: use existing env, fallback to .env for local dev
PROJECT_ROOT = Path(__file__).parent.parent
required_vars = ["SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET", "SPOTIFY_REFRESH_TOKEN"]
missing = [v for v in required_vars if not os.environ.get(v)]
if missing:
    load_dotenv(PROJECT_ROOT / ".env")

# Spotify API Configuration
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REFRESH_TOKEN = os.environ.get("SPOTIFY_REFRESH_TOKEN")

# Playlist Configuration
PLAYLIST_NAME = os.getenv("PROCESSING_PLAYLIST_NAME", "🎵 Choreography Queue")
PLAYLIST_DESCRIPTION = "Tracks awaiting choreography generation - managed automatically"
MAX_PLAYLIST_SIZE = int(os.getenv("MAX_PLAYLIST_SIZE", "50"))


def get_user_id(token):
    """Get current user's Spotify ID."""
    data = spotify_get(token, "https://api.spotify.com/v1/me")
    return data["id"]


def find_processing_playlist(token, user_id):
    """Find the processing queue playlist, or return None."""
    offset = 0
    limit = 50

    while True:
        data = spotify_get(
            token,
            f"https://api.spotify.com/v1/users/{user_id}/playlists",
            params={"limit": limit, "offset": offset},
        )

        for playlist in data.get("items", []):
            if playlist.get("name") == PLAYLIST_NAME:
                return playlist

        if not data.get("next"):
            break

        offset += limit

    return None


def create_processing_playlist(token, user_id):
    """Create a new processing queue playlist."""
    data = spotify_post(
        token,
        f"https://api.spotify.com/v1/users/{user_id}/playlists",
        data={
            "name": PLAYLIST_NAME,
            "description": PLAYLIST_DESCRIPTION,
            "public": False,
        },
    )
    return data


def get_playlist_tracks(token, playlist_id):
    """Get all track IDs from a playlist."""
    track_ids = []
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"

    while url:
        data = spotify_get(token, url, params={"fields": "items(track(id)),next"})

        for item in data.get("items", []):
            track = item.get("track")
            if track and track.get("id"):
                track_ids.append(track["id"])

        url = data.get("next")

    return track_ids


def add_tracks_to_playlist(token, playlist_id, track_uris):
    """Add tracks to playlist (max 100 at a time)."""
    if not track_uris:
        return

    # Spotify API limit: 100 tracks per request
    chunk_size = 100
    for i in range(0, len(track_uris), chunk_size):
        chunk = track_uris[i : i + chunk_size]
        spotify_post(
            token,
            f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks",
            data={"uris": chunk},
        )


def remove_tracks_from_playlist(token, playlist_id, track_uris):
    """Remove tracks from playlist."""
    if not track_uris:
        return

    # Spotify API limit: 100 tracks per request
    chunk_size = 100
    for i in range(0, len(track_uris), chunk_size):
        chunk = track_uris[i : i + chunk_size]
        tracks = [{"uri": uri} for uri in chunk]
        spotify_delete(
            token,
            f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks",
            data={"tracks": tracks},
        )


def replace_playlist_tracks(token, playlist_id, track_uris):
    """Replace all tracks in playlist with new ones."""
    # First, clear the playlist
    spotify_put(
        token,
        f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks",
        data={"uris": []},
    )

    # Then add new tracks
    add_tracks_to_playlist(token, playlist_id, track_uris)


def sync_playlist():
    """
    Main sync function:
    1. Get tracks from Base44 needing choreography
    2. Find or create processing playlist
    3. Update playlist with tracks to process
    4. Remove tracks that have been completed
    """
    print(f"\n{'='*60}")
    print(f"Managing Processing Playlist: {PLAYLIST_NAME}")
    print(f"{'='*60}\n")

    # Get Spotify token
    print("Step 1: Authenticating with Spotify...")
    token = refresh_access_token(
        SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REFRESH_TOKEN
    )
    user_id = get_user_id(token)
    print(f"  ✓ Logged in as: {user_id}\n")

    # Find or create playlist
    print("Step 2: Finding/creating processing playlist...")
    playlist = find_processing_playlist(token, user_id)

    if playlist:
        playlist_id = playlist["id"]
        print(f"  ✓ Found existing playlist: {playlist['external_urls']['spotify']}")
    else:
        playlist = create_processing_playlist(token, user_id)
        playlist_id = playlist["id"]
        print(f"  ✓ Created new playlist: {playlist['external_urls']['spotify']}")

    # Get current playlist contents
    print("\nStep 3: Getting current playlist tracks...")
    current_track_ids = get_playlist_tracks(token, playlist_id)
    print(f"  ✓ Current playlist has {len(current_track_ids)} tracks\n")

    # Get tracks from Base44 needing choreography
    print("Step 4: Fetching tracks from Base44...")
    all_base44_tracks = get_all_tracks()
    tracks_needing_choreo = get_track_spotify_ids_needing_choreography(
        all_base44_tracks
    )
    print(f"  ✓ Found {len(tracks_needing_choreo)} tracks needing choreography\n")

    # Calculate changes
    current_set = set(current_track_ids)
    needed_set = set(tracks_needing_choreo)

    # Tracks to remove (completed or no longer needed)
    to_remove = current_set - needed_set

    # Tracks to add (new tracks needing choreography)
    to_add = needed_set - current_set

    # Limit playlist size
    if len(needed_set) > MAX_PLAYLIST_SIZE:
        # Take only the first N tracks (already limited by needed_set)
        limited_set = set(list(needed_set)[:MAX_PLAYLIST_SIZE])
        to_add = limited_set - current_set
        print(
            f"  ℹ️  Limiting playlist to {MAX_PLAYLIST_SIZE} tracks "
            f"(total available: {len(needed_set)})\n"
        )

    # Apply changes
    print("Step 5: Updating playlist...")

    if to_remove:
        remove_uris = [f"spotify:track:{tid}" for tid in to_remove]
        remove_tracks_from_playlist(token, playlist_id, remove_uris)
        print(f"  ✓ Removed {len(to_remove)} completed tracks")

    if to_add:
        add_uris = [f"spotify:track:{tid}" for tid in list(to_add)[:MAX_PLAYLIST_SIZE]]
        add_tracks_to_playlist(token, playlist_id, add_uris)
        print(f"  ✓ Added {len(add_uris)} new tracks")

    if not to_remove and not to_add:
        print("  ✓ Playlist is up to date - no changes needed")

    # Final summary
    final_count = len(current_track_ids) - len(to_remove) + len(to_add)

    print(f"\n{'='*60}")
    print("Playlist Update Complete")
    print(f"{'='*60}")
    print(f"Playlist: {PLAYLIST_NAME}")
    print(f"URL: {playlist['external_urls']['spotify']}")
    print(f"Total tracks: {final_count}")
    print(f"Removed: {len(to_remove)}")
    print(f"Added: {len(to_add)}")
    print(f"Remaining to process: {len(tracks_needing_choreo)}")

    if final_count > 0:
        print("\n✅ Ready to capture! Run:")
        print("   python capture/auto_capture_playlist_only.py")
        print(
            f"\n   Set in .env: SPOTIFY_PLAYLIST_URL={playlist['external_urls']['spotify']}"
        )


def main():
    """Main entry point."""
    # Re-fetch after possible .env load
    SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID")
    SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET")
    SPOTIFY_REFRESH_TOKEN = os.environ.get("SPOTIFY_REFRESH_TOKEN")
    if not all([SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REFRESH_TOKEN]):
        print("❌ Missing Spotify credentials in environment or .env file")
        print("   Required: SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REFRESH_TOKEN")
        sys.exit(1)

    sync_playlist()


if __name__ == "__main__":
    main()
