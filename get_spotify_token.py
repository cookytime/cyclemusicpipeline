#!/usr/bin/env python3
"""
Helper script to get Spotify refresh token using OAuth flow.

Usage:
    1. Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in your environment or .env
    2. Run: python get_spotify_token.py
    3. Follow the URL, authorize the app, copy the redirect URL
    4. Paste the full redirect URL back into the script
    5. Your refresh token will be displayed
"""

import os
import sys

import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth

# Load environment variables
load_dotenv()

# Get credentials from environment
CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    print("❌ Error: SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set")
    print("\nSet them in your .env file or environment:")
    print("  SPOTIFY_CLIENT_ID=your_client_id")
    print("  SPOTIFY_CLIENT_SECRET=your_client_secret")
    sys.exit(1)

# Define the scopes needed for your application
SCOPES = [
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
    "playlist-read-private",
    "playlist-read-collaborative",
    "user-library-read",
]

# Use explicit IP addresses as required by Spotify (not "localhost")
REDIRECT_URIS = [
    "http://127.0.0.1:8888/callback",  # IPv4 loopback - RECOMMENDED
    "http://[::1]:8888/callback",  # IPv6 loopback
    "http://127.0.0.1:3000/callback",
]

# Use the first one by default (127.0.0.1)
REDIRECT_URI = REDIRECT_URIS[0]

print("🎵 Spotify Refresh Token Generator")
print("=" * 50)
print()
print("⚠️  IMPORTANT: First configure your Spotify App Dashboard:")
print("   1. Go to: https://developer.spotify.com/dashboard")
print("   2. Select your app (or create one)")
print("   3. Click 'Settings' or 'Edit Settings'")
print("   4. Add this EXACT Redirect URI (use the IP, not 'localhost'):")
print(f"      → {REDIRECT_URI}")
print("   5. Click 'ADD', then scroll down and click 'SAVE'")
print()
print("⚠️  Must use 127.0.0.1 (not localhost) - Spotify security requirement!")
print()
input("Press Enter when you've added the redirect URI in the dashboard...")

# Create OAuth handler
sp_oauth = SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=" ".join(SCOPES),
    open_browser=True,  # Will try to open browser automatically
)

# Get the authorization URL
auth_url = sp_oauth.get_authorize_url()

print("Step 1: Opening authorization URL in your browser...")
print(f"If it doesn't open automatically, go to:\n{auth_url}\n")

print("Step 2: After authorizing, you'll be redirected to a URL like:")
print("  http://127.0.0.1:8888/callback?code=AQD...")
print()
print("⚠️  The page will show an error (can't connect) - that's OK!")
print("   Just copy the ENTIRE URL from your browser's address bar.")
print(
    "   (The authorization code can only be used ONCE - get a fresh one if this fails)"
)
print()

# Get the redirect URL from user
response_url = input("Paste the full redirect URL here: ").strip()

try:
    # Extract the authorization code and get the token
    code = sp_oauth.parse_response_code(response_url)
    token_info = sp_oauth.get_access_token(code, check_cache=False)

    print("\n" + "=" * 50)
    print("✅ Success! Here are your tokens:")
    print("=" * 50)
    print()
    print("Access Token (expires in 1 hour):")
    print(token_info["access_token"])
    print()
    print("Refresh Token (use this in your .env file):")
    print(token_info["refresh_token"])
    print()
    print("=" * 50)
    print()
    print("Add this to your .env file:")
    print(f'SPOTIFY_REFRESH_TOKEN={token_info["refresh_token"]}')
    print()

    # Test the token
    print("Testing the token...")
    sp = spotipy.Spotify(auth=token_info["access_token"])
    user = sp.current_user()
    if user is not None and "display_name" in user and "id" in user:
        print(f"✅ Token works! Logged in as: {user['display_name']} ({user['id']})")
    else:
        print("⚠️  Token test failed: Could not retrieve user information.")

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("  1. Make sure you copied the ENTIRE URL from the browser")
    print("  2. Check the redirect URI in Spotify Developer Dashboard matches exactly:")
    print(f"     → Current redirect URI: {REDIRECT_URI}")
    print(f"     → Dashboard: https://developer.spotify.com/dashboard")
    print("  3. The 'not secure' warning is normal for localhost - just click through")
    print("  4. Make sure you clicked 'Agree' to authorize the app")
    print("\nAlternative: Try changing REDIRECT_URI in this script to:")
    for uri in REDIRECT_URIS:
        if uri != REDIRECT_URI:
            print(f"  {uri}")
    sys.exit(1)
