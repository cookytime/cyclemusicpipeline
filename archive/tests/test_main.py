import pytest

def test_extract_spotify_id():
    from main import extract_spotify_id
    # Valid Spotify ID (22 chars)
    assert extract_spotify_id("2grjqo0Frpf2okIBiifQKs.wav") == "2grjqo0Frpf2okIBiifQKs"
    # Invalid filename
    assert extract_spotify_id("not_a_spotify_file.wav") is None
    # Embedded in string
    assert extract_spotify_id("foo_2grjqo0Frpf2okIBiifQKs_bar.wav") == "2grjqo0Frpf2okIBiifQKs"

# Add more tests for other utility functions as needed
