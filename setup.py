from setuptools import setup, find_packages

setup(
    name="cyclemusicpipeline",
    version="0.1.0",
    description="Automated Spotify playlist capture, analysis, and choreography pipeline.",
    author="cookytime",
    packages=find_packages(include=["analyze*", "manage*", "capture*", "archive*", "prompts*"]),
    install_requires=[
        "requests",
        "python-dotenv",
        "tqdm",
        # Add any other Python dependencies here
    ],
    entry_points={
        "console_scripts": [
            "auto-capture-playlist=capture.auto_capture_playlist_only:main",
            # Add more CLI entry points as needed
        ]
    },
    include_package_data=True,
    python_requires=">=3.8",
)