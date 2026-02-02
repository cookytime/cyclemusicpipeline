from __future__ import annotations

from typing import Any

import requests


def refresh_access_token(
    client_id: str,
    client_secret: str,
    refresh_token: str,
    timeout: int = 10,
) -> str:
    """Refresh Spotify access token."""
    response = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def spotify_get(
    token: str, url: str, params: dict[str, Any] | None = None, timeout: int = 10
) -> dict[str, Any]:
    """Make a GET request to the Spotify API."""
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        params=params,
        timeout=timeout,
    )
    if response.status_code == 401:
        raise PermissionError("Spotify token expired/unauthorized")
    response.raise_for_status()
    return response.json()


def spotify_post(
    token: str, url: str, data: dict[str, Any] | None = None, timeout: int = 10
) -> dict[str, Any] | None:
    """Make a POST request to the Spotify API."""
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=data,
        timeout=timeout,
    )
    if response.status_code == 401:
        raise PermissionError("Spotify token expired/unauthorized")
    response.raise_for_status()
    return response.json() if response.text else None


def spotify_put(
    token: str, url: str, data: dict[str, Any] | None = None, timeout: int = 10
) -> dict[str, Any] | None:
    """Make a PUT request to the Spotify API."""
    response = requests.put(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=data,
        timeout=timeout,
    )
    if response.status_code == 401:
        raise PermissionError("Spotify token expired/unauthorized")
    response.raise_for_status()
    return response.json() if response.text else None


def spotify_delete(
    token: str, url: str, data: dict[str, Any] | None = None, timeout: int = 10
) -> None:
    """Make a DELETE request to the Spotify API."""
    response = requests.delete(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=data,
        timeout=timeout,
    )
    if response.status_code == 401:
        raise PermissionError("Spotify token expired/unauthorized")
    response.raise_for_status()


def spotify_get_currently_playing(
    token: str, timeout: int = 10
) -> dict[str, Any] | None:
    """Get the user's currently playing track, returning None when idle."""
    response = requests.get(
        "https://api.spotify.com/v1/me/player/currently-playing",
        headers={"Authorization": f"Bearer {token}"},
        timeout=timeout,
    )
    if response.status_code == 204:
        return None
    if response.status_code == 401:
        raise PermissionError("Spotify token expired/unauthorized")
    response.raise_for_status()
    return response.json()
