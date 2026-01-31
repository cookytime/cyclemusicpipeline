import json
import os
import re
import sys
from pathlib import Path
from string import Template

import librosa
import librosa.beat
import librosa.effects
import librosa.segment
import numpy as np
from dotenv import load_dotenv
from mutagen import File as MutagenFile
from openai import OpenAI


# Suggestion 7: Safe JSON Serialization for Numpy Types
class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        return super(NumpyEncoder, self).default(o)


# -------------------------------------------------------------------
# OpenAI choreography generator (schema enforced)
# -------------------------------------------------------------------


PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"  # project root /prompts


def load_prompt(name: str) -> str:
    p = PROMPTS_DIR / name
    return p.read_text(encoding="utf-8")


def render_user_prompt(
    rider_settings: dict, music_map: dict, allowed_block_starts: list[str]
) -> str:
    raw = load_prompt("user.txt")
    allowed = ", ".join(allowed_block_starts)
    return Template(raw).safe_substitute(
        rider_settings=json.dumps(rider_settings, indent=2),
        music_map=json.dumps(music_map, cls=NumpyEncoder, indent=2),
        allowed_block_starts=allowed,
    )


def generate_track_choreography_openai(music_map: dict, rider_settings: dict) -> dict:
    """Generate a Track JSON object matching the Track schema."""
    client = OpenAI()

    track_schema = {
        "name": "Track",
        "type": "object",
        "properties": {
            "title": {
                "type": ["string", "null"],
                "description": "Name of the track/song",
            },
            "artist": {"type": ["string", "null"], "description": "Artist name"},
            "album": {"type": ["string", "null"], "description": "Album name"},
            "spotify_id": {
                "type": ["string", "null"],
                "description": "Spotify track ID",
            },
            "spotify_album_art": {
                "type": ["string", "null"],
                "description": "URL to Spotify album artwork",
            },
            "spotify_url": {
                "type": ["string", "null"],
                "description": "Spotify track URL for playback",
            },
            "duration_minutes": {
                "type": ["number", "null"],
                "description": "Duration in minutes",
            },
            "bpm": {"type": ["number", "null"], "description": "Beats per minute"},
            "intensity": {
                "type": ["string", "null"],
                "enum": ["low", "medium", "high", "extreme", None],
                "description": "Intensity level of the track",
            },
            "focus_area": {
                "type": ["string", "null"],
                "enum": [
                    "warmup",
                    "climb",
                    "sprint",
                    "recovery",
                    "cooldown",
                    "intervals",
                    "endurance",
                    None,
                ],
                "description": "Main focus of this track",
            },
            "track_type": {
                "type": ["string", "null"],
                "description": "Type of ride (e.g., Rhythm Ride, Speed Intervals, Strength Climb)",
            },
            "position": {
                "type": ["string", "null"],
                "description": "Primary position (e.g., Saddle, Standing, Saddle → Standing)",
            },
            "resistance_min": {
                "type": ["number", "null"],
                "description": "Minimum resistance level",
            },
            "resistance_max": {
                "type": ["number", "null"],
                "description": "Maximum resistance level",
            },
            "cadence_min": {
                "type": ["number", "null"],
                "description": "Minimum cadence (rpm)",
            },
            "cadence_max": {
                "type": ["number", "null"],
                "description": "Maximum cadence (rpm)",
            },
            "choreography": {
                "type": "array",
                "description": "Choreography moves/cues for the track",
                "items": {
                    "type": "object",
                    "properties": {
                        "timestamp": {
                            "type": ["string", "null"],
                            "description": "When to start this move (e.g., 0:30)",
                        },
                        "move": {
                            "type": ["string", "null"],
                            "description": "The move or cue",
                        },
                        "duration_seconds": {
                            "type": ["number", "null"],
                            "description": "How long to hold this move",
                        },
                        "notes": {
                            "type": ["string", "null"],
                            "description": "Additional coaching notes",
                        },
                        "type": {
                            "type": ["string", "null"],
                            "description": "Type of cue",
                        },
                    },
                    "required": [
                        "timestamp",
                        "move",
                        "duration_seconds",
                        "notes",
                        "type",
                    ],
                    "additionalProperties": False,
                },
            },
            "cues": {
                "type": "array",
                "description": "Coaching cues and motivational phrases",
                "items": {"type": "string"},
            },
            "notes": {
                "type": ["string", "null"],
                "description": "General notes about the track",
            },
        },
        "required": [
            "title",
            "artist",
            "album",
            "spotify_id",
            "spotify_album_art",
            "spotify_url",
            "duration_minutes",
            "bpm",
            "intensity",
            "focus_area",
            "track_type",
            "position",
            "resistance_min",
            "resistance_max",
            "cadence_min",
            "cadence_max",
            "choreography",
            "cues",
            "notes",
        ],
        "additionalProperties": False,
    }

    system_text = load_prompt("system.txt")
    allowed_starts = build_allowed_block_starts(music_map)
    user_text = render_user_prompt(rider_settings, music_map, allowed_starts)

    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "Track", "schema": track_schema, "strict": True},
        },
    )

    out_text = resp.choices[0].message.content
    track_data = json.loads(out_text)

    # Force accurate duration from music_map
    dur_s = float((music_map.get("metadata") or {}).get("duration_s") or 0.0)
    if dur_s > 0:
        track_data["duration_minutes"] = round(dur_s / 60.0, 2)

    # Clean timestamps and durations (use duration_s from music_map)
    track_data = clean_choreography_timestamps(
        track_data, duration_s_override=dur_s if dur_s > 0 else None
    )

    return track_data


# Helper functions for timestamp cleaning

_TS_RE = re.compile(r"^\s*(\d+):(\d+)\s*$")


def ts_to_seconds_loose(ts: str) -> int | None:
    """
    Accepts M:SS, but also fixes cases like 2:62 by carrying seconds into minutes.
    Returns seconds as int, or None if unparseable.
    """
    m = _TS_RE.match(ts or "")
    if not m:
        return None
    mm = int(m.group(1))
    ss = int(m.group(2))
    mm += ss // 60
    ss = ss % 60
    return mm * 60 + ss


def seconds_to_ts(sec: int) -> str:
    mm = sec // 60
    ss = sec % 60
    return f"{mm}:{ss:02d}"


from bisect import bisect_left


def snap_to_next_downbeat(t: float, downbeats: list[float]) -> float:
    """Snap time to the nearest downbeat at or after t."""
    if not downbeats:
        return float(t)
    i = bisect_left(downbeats, t - 1e-6)
    if i >= len(downbeats):
        return float(downbeats[-1])
    return float(downbeats[i])


def build_allowed_block_starts(
    music_map: dict,
    first_within_s: float = 10.0,
    last_within_s: float = 20.0,
    target_gap_s: float = 34.0,
    max_gap_s: float = 55.0,
) -> list[str]:
    """
    Compute REV-style candidate block starts as DOWNBEATS only.
    Uses timeline boundaries, anchor times, and fills long gaps.
    Returns timestamps in M:SS.
    """
    duration_s = float((music_map.get("metadata") or {}).get("duration_s") or 0.0)
    downbeats = list((music_map.get("global") or {}).get("downbeats_s") or [])
    timeline = list(music_map.get("timeline") or [])
    anchors = list(music_map.get("anchors") or [])

    if duration_s <= 0 and downbeats:
        duration_s = float(downbeats[-1])
    if duration_s <= 0:
        duration_s = 240.0

    candidates = set()

    # Coverage endpoints
    candidates.add(snap_to_next_downbeat(0.0, downbeats))
    candidates.add(snap_to_next_downbeat(min(first_within_s, duration_s), downbeats))
    candidates.add(
        snap_to_next_downbeat(max(0.0, duration_s - last_within_s), downbeats)
    )

    # Timeline boundaries + midpoint for long segments
    for seg in timeline:
        start = float(seg.get("start_s") or 0.0)
        end = float(seg.get("end_s") or start)
        candidates.add(snap_to_next_downbeat(start, downbeats))
        if end - start > 50.0:
            mid = start + (end - start) / 2.0
            candidates.add(snap_to_next_downbeat(mid, downbeats))

    # Anchors (drops/peaks)
    for a in anchors:
        t = float(a.get("time_s") or 0.0)
        candidates.add(snap_to_next_downbeat(t, downbeats))

    cand = sorted(t for t in candidates if 0.0 <= t <= duration_s)

    # Fill large gaps
    filled = [cand[0]] if cand else [snap_to_next_downbeat(0.0, downbeats)]
    for t in cand[1:]:
        prev = filled[-1]
        gap = t - prev
        while gap > max_gap_s:
            new_t = snap_to_next_downbeat(prev + target_gap_s, downbeats)
            if new_t <= prev + 0.5:
                break
            filled.append(new_t)
            prev = new_t
            gap = t - prev
        filled.append(t)

    # Dedup and sort
    dedup = []
    for t in sorted(set(filled)):
        if not dedup or abs(t - dedup[-1]) > 0.75:
            dedup.append(t)

    # Target count by duration
    if duration_s < 180:
        desired_min, desired_max = 5, 6
    elif duration_s < 240:
        desired_min, desired_max = 6, 8
    else:
        desired_min, desired_max = 7, 9

    # Downselect if too many (keep first/last)
    if len(dedup) > desired_max:
        first = dedup[0]
        last = dedup[-1]
        middle = dedup[1:-1]
        keep_mid = max(0, desired_max - 2)
        if keep_mid <= 0:
            selected = [first, last]
        else:
            if keep_mid == 1:
                selected_mid = [middle[len(middle) // 2]] if middle else []
            else:
                idxs = [
                    round(i * (len(middle) - 1) / (keep_mid - 1))
                    for i in range(keep_mid)
                ]
                selected_mid = [
                    middle[int(i)] for i in sorted(set(int(x) for x in idxs)) if middle
                ]
            selected = [first] + selected_mid + [last]
        dedup = selected

    # Add midpoints if too few
    while len(dedup) < desired_min and len(dedup) >= 2:
        gaps = [(dedup[i + 1] - dedup[i], i) for i in range(len(dedup) - 1)]
        gaps.sort(reverse=True)
        gap, i = gaps[0]
        insert_t = snap_to_next_downbeat(dedup[i] + gap / 2.0, downbeats)
        if insert_t <= dedup[i] + 0.5 or insert_t >= dedup[i + 1] - 0.5:
            break
        dedup = sorted(set(dedup + [insert_t]))

    return [seconds_to_ts(int(round(t))) for t in dedup]


def clean_choreography_timestamps(
    track: dict, duration_s_override: float | None = None
) -> dict:
    duration_s = int(round(float(duration_s_override or 0)))
    if duration_s <= 0:
        duration_s = int(round(float(track.get("duration_minutes", 0) or 0) * 60))
    if duration_s <= 0:
        duration_s = 10_000

    cleaned = []
    for item in track.get("choreography", []) or []:
        raw_ts = item.get("timestamp")
        sec = ts_to_seconds_loose(raw_ts)
        if sec is None:
            continue
        if sec > duration_s:
            continue
        item = dict(item)
        item["timestamp"] = seconds_to_ts(sec)
        cleaned.append(item)

    cleaned.sort(key=lambda x: ts_to_seconds_loose(x["timestamp"]) or 0)
    for i, item in enumerate(cleaned):
        t0 = ts_to_seconds_loose(item["timestamp"]) or 0
        t1 = (
            ts_to_seconds_loose(cleaned[i + 1]["timestamp"])
            if i + 1 < len(cleaned)
            else duration_s
        )
        if t1 is None:
            t1 = duration_s
        item["duration_seconds"] = max(10, min(90, int(t1 - t0)))

    track = dict(track)
    track["choreography"] = cleaned
    return track


class TrackAnalyzer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.y = None
        self.sr = 22050  # Suggestion 2: Standardize SR for performance
        self.duration = 0.0
        self.hop_length = 512

        # Suggestion 4: Track silence offset to keep timestamps aligned with original file
        self.time_offset = 0.0

        # Features
        self.rms = None
        self.spectral_centroid = None
        self.onset_env = None
        self.chroma = None  # Needed for Key detection

        # Beat / bar timing
        self.tempo = 0.0
        self.beats = None
        self.beat_times = None
        self.downbeat_times = None

        # Musical Data
        self.key = None  # Suggestion 5

        # Anchor tuning
        self.max_anchors_per_min = 2.0
        self.min_anchor_spacing_s = 6.0
        self.snap_anchors_to_downbeat = True
        self.snap_tolerance_s = 0.15  # Suggestion 6: Only snap if within 150ms
        self.drop_priority_bonus = 0.25

    def _to_global_time(self, t):
        """Helper to convert analysis time back to original file time (accounting for trim)."""
        if isinstance(t, (list, np.ndarray)):
            return [float(x) + self.time_offset for x in t]
        return float(t) + self.time_offset

    def load_audio(self):
        """Loads audio, trims silence, extracts features, and aligns grid."""
        try:
            print("Loading audio file...")
            # Suggestion 2: Fixed SR
            y_raw, self.sr = librosa.load(self.file_path, sr=self.sr, mono=True)

            # Suggestion 4: Trim leading/trailing silence for accurate analysis
            print("Trimming silence...")
            self.y, trim_indices = librosa.effects.trim(y_raw, top_db=30)
            self.time_offset = trim_indices[0] / self.sr  # Calculate start offset

            self.duration = float(librosa.get_duration(y=self.y, sr=self.sr))
            print(
                f"Analysis Duration: {self.duration:.2f}s (Offset: {self.time_offset:.2f}s)"
            )

            print("Computing features (RMS, Centroid, Onset)...")
            self.rms = librosa.feature.rms(y=self.y, hop_length=self.hop_length)[0]

            self.spectral_centroid = librosa.feature.spectral_centroid(
                y=self.y, sr=self.sr, hop_length=self.hop_length
            )[0]

            self.onset_env = librosa.onset.onset_strength(
                y=self.y, sr=self.sr, hop_length=self.hop_length
            )

            # Suggestion 5: Compute Chroma for Key Detection and Segmentation
            self.chroma = librosa.feature.chroma_cqt(
                y=self.y, sr=self.sr, hop_length=self.hop_length
            )

            # Beat tracking
            print("Extracting beat grid...")
            self.tempo, self.beats = librosa.beat.beat_track(
                onset_envelope=self.onset_env,
                sr=self.sr,
                hop_length=self.hop_length,
            )
            self.tempo = float(self.tempo)

            # Local beat times (relative to trimmed audio)
            local_beat_times = librosa.frames_to_time(
                self.beats, sr=self.sr, hop_length=self.hop_length
            )

            # Suggestion 3: Energy-Based Downbeat Phase Alignment
            # Find the loudest beat in the first measure to assume it's the "1"
            if len(self.beats) > 4:
                # Check RMS energy at the first 4 beat frames
                candidates = self.beats[:4]
                # Ensure we don't go out of bounds
                candidates = candidates[candidates < len(self.rms)]
                if len(candidates) > 0:
                    energies = self.rms[candidates]
                    phase_offset = np.argmax(
                        energies
                    )  # Index of the loudest beat (0, 1, 2, or 3)
                else:
                    phase_offset = 0
            else:
                phase_offset = 0

            # Apply phase offset to downbeats
            local_downbeats = local_beat_times[phase_offset::4]

            # Convert to global times (original file timeline)
            self.beat_times = self._to_global_time(local_beat_times)
            self.downbeat_times = self._to_global_time(local_downbeats)

            print("Audio loading complete.")
        except Exception as e:
            print(f"Error loading audio: {e}")
            raise

    def detect_key(self):
        """Suggestion 5: Detect Musical Key using Chroma correlation."""
        try:
            # Krumhansl-Schmuckler key profiles
            maj_profile = np.array(
                [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
            )
            min_profile = np.array(
                [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
            )

            chroma_mean = np.mean(self.chroma, axis=1)

            # Correlate
            maj_corrs = [
                np.corrcoef(chroma_mean, np.roll(maj_profile, i))[0, 1]
                for i in range(12)
            ]
            min_corrs = [
                np.corrcoef(chroma_mean, np.roll(min_profile, i))[0, 1]
                for i in range(12)
            ]

            max_maj = np.max(maj_corrs)
            max_min = np.max(min_corrs)

            pitch_classes = [
                "C",
                "C#",
                "D",
                "D#",
                "E",
                "F",
                "F#",
                "G",
                "G#",
                "A",
                "A#",
                "B",
            ]

            if max_maj > max_min:
                key_idx = np.argmax(maj_corrs)
                self.key = f"{pitch_classes[key_idx]} Major"
            else:
                key_idx = np.argmax(min_corrs)
                self.key = f"{pitch_classes[key_idx]} Minor"

        except Exception as e:
            print(f"Key detection failed: {e}")
            self.key = "Unknown"

    def get_metadata(self):
        """Extracts file tags and technical specs."""
        self.detect_key()  # Run key detection

        meta = {
            "title": None,
            "artist": None,
            "duration_s": round(float(self.duration), 2),
            "sample_rate": int(self.sr),
            "bpm": round(float(self.tempo), 1),
            "key": self.key,
            "bpm_confidence": 0.0,
        }

        # Suggestion 1: Use mutagen.File for generic tag support (MP3, FLAC, etc)
        try:
            audio_tags = MutagenFile(self.file_path)
            if audio_tags:
                # Try standard ID3/Vorbis keys
                # EasyID3/MP3 often uses 'TIT2'/'TPE1', Ogg uses 'TITLE'/'ARTIST'
                # We check common keys loosely
                tags = audio_tags.tags
                if tags:
                    # Helper to find tag case-insensitively
                    def get_tag(keys):
                        for k in keys:
                            if k in tags:
                                val = tags[k]
                                return (
                                    str(val[0]) if isinstance(val, list) else str(val)
                                )
                        return None

                    meta["title"] = get_tag(
                        ["TIT2", "title", "TITLE"]
                    ) or os.path.basename(self.file_path)
                    meta["artist"] = get_tag(["TPE1", "artist", "ARTIST"]) or "Unknown"
        except Exception as e:
            print(f"Metadata extraction warning: {e}")

        # Confidence estimate
        try:
            tempogram = librosa.feature.tempogram(
                onset_envelope=self.onset_env, sr=self.sr
            )
            meta["bpm_confidence"] = round(float(np.max(np.mean(tempogram, axis=1))), 2)
        except Exception:
            meta["bpm_confidence"] = 0.0

        return meta

    @staticmethod
    def _snap_to_nearest(t: float, grid: np.ndarray, threshold: float = 0.15) -> float:
        """
        Suggestion 6: Snap time to nearest grid time ONLY if within threshold.
        """
        if grid is None or len(grid) == 0:
            return t

        # Find nearest index
        idx = int(np.argmin(np.abs(grid - t)))
        nearest_val = float(grid[idx])
        dist = abs(nearest_val - t)

        if dist <= threshold:
            return nearest_val
        return t

    @staticmethod
    def _enforce_spacing(anchors, min_spacing_s: float):
        if not anchors:
            return anchors

        anchors = sorted(anchors, key=lambda a: a["time_s"])
        kept = []

        for a in anchors:
            if not kept:
                kept.append(a)
                continue

            if a["time_s"] - kept[-1]["time_s"] >= min_spacing_s:
                kept.append(a)
                continue

            prev = kept[-1]
            if a["importance"] > prev["importance"]:
                kept[-1] = a
            elif a["importance"] == prev["importance"]:
                if a["type"] == "drop" and prev["type"] != "drop":
                    kept[-1] = a

        return kept

    def get_segmentation(self):
        """Breaks track into musical segments."""
        # Use pre-computed chroma
        chroma_stack = librosa.feature.stack_memory(self.chroma, n_steps=10, delay=3)

        bounds_frames = librosa.segment.agglomerative(chroma_stack, k=8)
        local_bounds_times = librosa.frames_to_time(
            bounds_frames, sr=self.sr, hop_length=self.hop_length
        )

        local_bounds_times = np.unique(
            np.concatenate(([0.0], local_bounds_times, [self.duration]))
        )
        local_bounds_times.sort()

        segments = []
        max_rms = float(np.max(self.rms) + 1e-6)
        max_cent = float(np.max(self.spectral_centroid) + 1e-6)

        for i in range(len(local_bounds_times) - 1):
            start_local = float(local_bounds_times[i])
            end_local = float(local_bounds_times[i + 1])

            f_start = int(
                librosa.time_to_frames(
                    start_local, sr=self.sr, hop_length=self.hop_length
                )
            )
            f_end = int(
                librosa.time_to_frames(
                    end_local, sr=self.sr, hop_length=self.hop_length
                )
            )
            f_end = max(f_end, f_start + 1)

            seg_rms = float(np.mean(self.rms[f_start:f_end]))
            seg_cent = float(np.mean(self.spectral_centroid[f_start:f_end]))

            energy = round(seg_rms / max_rms, 2)
            intensity = round(seg_cent / max_cent, 2)
            tension = round(float(energy * intensity), 2)

            intent_hint = None
            if start_local < 15.0 and energy < 0.4:
                intent_hint = "intro"
            elif end_local > (self.duration - 15.0) and energy < 0.4:
                intent_hint = "outro"
            else:
                if energy >= 0.75:
                    intent_hint = "surge"
                elif energy >= 0.4:
                    slope, _ = np.polyfit(
                        np.arange(f_end - f_start), self.rms[f_start:f_end], 1
                    )
                    intent_hint = "build" if slope > 0.0001 else "steady"
                else:
                    intent_hint = "steady"

            # Convert local times to global times for output
            global_start = self._to_global_time(start_local)
            global_end = self._to_global_time(end_local)

            # Segment downbeats
            seg_downbeats = []
            if self.downbeat_times is not None:
                # Filter global downbeats
                seg_downbeats = [
                    round(float(t), 2)
                    for t in self.downbeat_times
                    if global_start <= t <= global_end
                ]

            segments.append(
                {
                    "start_s": round(global_start, 2),
                    "end_s": round(global_end, 2),
                    "energy": energy,
                    "intensity": intensity,
                    "tension": tension,
                    "intent_hint": intent_hint,
                    "downbeats_s": seg_downbeats,
                }
            )

        return segments

    def get_anchors(self):
        """Identifies choreography-friendly cue points."""
        anchors = []

        peak_frames = librosa.util.peak_pick(
            self.onset_env,
            pre_max=20,
            post_max=20,
            pre_avg=20,
            post_avg=20,
            delta=0.5,
            wait=20,
        )
        local_peak_times = librosa.frames_to_time(
            peak_frames, sr=self.sr, hop_length=self.hop_length
        )

        rms_diff = np.diff(self.rms)
        if len(rms_diff) > 0:
            thresh = float(np.max(rms_diff) * 0.7)
            drop_frames = np.where(rms_diff > thresh)[0]
        else:
            drop_frames = np.array([], dtype=int)

        local_drop_times = librosa.frames_to_time(
            drop_frames, sr=self.sr, hop_length=self.hop_length
        )

        # Process Drops
        for t_local in local_drop_times:
            t_global = self._to_global_time(t_local)

            if self.snap_anchors_to_downbeat and self.downbeat_times is not None:
                # Suggestion 6: Use tolerance
                t_global = self._snap_to_nearest(
                    t_global, self.downbeat_times, self.snap_tolerance_s
                )

            anchors.append(
                {
                    "time_s": round(t_global, 2),
                    "type": "drop",
                    "confidence": 0.85,
                    "reason": "Sudden energy spike",
                }
            )

        # Process Peaks
        for t_local in local_peak_times:
            t_global = self._to_global_time(t_local)

            if self.snap_anchors_to_downbeat and self.downbeat_times is not None:
                t_global = self._snap_to_nearest(
                    t_global, self.downbeat_times, self.snap_tolerance_s
                )

            if not any(abs(t_global - a["time_s"]) < 1.0 for a in anchors):
                anchors.append(
                    {
                        "time_s": round(t_global, 2),
                        "type": "peak",
                        "confidence": 0.7,
                        "reason": "Strong transient onset",
                    }
                )

        # Deduplicate
        anchors.sort(key=lambda x: (x["time_s"], 0 if x["type"] == "drop" else 1))
        dedup = []
        seen = set()
        for a in anchors:
            key = a["time_s"]
            if key in seen:
                if a["type"] == "drop":
                    for i in range(len(dedup) - 1, -1, -1):
                        if dedup[i]["time_s"] == key:
                            dedup[i] = a
                            break
            else:
                dedup.append(a)
                seen.add(key)

        # Score importance
        for a in dedup:
            imp = float(a["confidence"])
            if a["type"] == "drop":
                imp += self.drop_priority_bonus
            a["importance"] = round(imp, 3)

        spaced = self._enforce_spacing(dedup, self.min_anchor_spacing_s)

        max_keep = max(8, int((self.duration / 60.0) * self.max_anchors_per_min))
        spaced_sorted = sorted(spaced, key=lambda a: a["importance"], reverse=True)[
            :max_keep
        ]
        spaced_sorted.sort(key=lambda a: a["time_s"])

        return spaced_sorted

    def load_spotify_metadata(self):
        base_path = os.path.splitext(self.file_path)[0]
        metadata_path = f"{base_path}.metadata.json"

        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(
                    f"Warning: Failed to load Spotify metadata from {metadata_path}: {e}"
                )
                return None
        return None

    def analyze(self):
        print(f"Analyzing {self.file_path} ...")
        self.load_audio()

        output = {
            "metadata": self.get_metadata(),
            "global": {
                "tempo_bpm": round(float(self.tempo), 1),
                "beats_s": (
                    [round(float(t), 2) for t in self.beat_times]
                    if self.beat_times is not None
                    else []
                ),
                "downbeats_s": (
                    [round(float(t), 2) for t in self.downbeat_times]
                    if self.downbeat_times is not None
                    else []
                ),
            },
            "timeline": self.get_segmentation(),
            "anchors": self.get_anchors(),
        }

        spotify_metadata = self.load_spotify_metadata()
        if spotify_metadata:
            print("Merging Spotify metadata...")
            output["spotify"] = spotify_metadata
            if spotify_metadata.get("name"):
                output["metadata"]["title"] = spotify_metadata["name"]
            if spotify_metadata.get("artists") and len(spotify_metadata["artists"]) > 0:
                output["metadata"]["artist"] = ", ".join(
                    a["name"] for a in spotify_metadata["artists"] if a.get("name")
                )

        return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_track.py <path_to_audio>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print("File not found.")
        sys.exit(1)

    analyzer = TrackAnalyzer(file_path)
    result = analyzer.analyze()

    # Write music_map.json to the same directory as the audio file
    base_path = os.path.splitext(file_path)[0]
    output_file = f"{base_path}.music_map.json"

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    print(f"Analysis complete. Output written to {output_file}")

    # -------------------------------------------------------------------
    # Generate choreography with OpenAI
    # -------------------------------------------------------------------
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found in .env - skipping choreography generation")
        sys.exit(0)

    rider_settings = {
        "rider_level": os.environ.get("RIDER_LEVEL", "intermediate"),
        "resistance_scale": {"min": 1, "max": 24},
        "cadence_limits": {
            "seated": {"min_rpm": 60, "max_rpm": 115},
            "standing": {"min_rpm": 55, "max_rpm": 80},
        },
        "cue_spacing_s": {"min": 24, "max": 32},
    }

    try:
        print("\nGenerating choreography with OpenAI...")
        track_json = generate_track_choreography_openai(result, rider_settings)

        choreography_path = f"{base_path}.choreography.json"
        output_data = {"track": track_json}

        with open(choreography_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"✓ Choreography saved: {choreography_path}")

        # Also save with spotify_id filename if different
        spotify_id = track_json.get("spotify_id") or result.get("spotify", {}).get(
            "spotify_id"
        )
        if spotify_id:
            captures_dir = os.path.dirname(os.path.abspath(file_path))
            spotify_named_path = os.path.join(
                captures_dir, f"{spotify_id}.choreography.json"
            )

            if os.path.abspath(spotify_named_path) != os.path.abspath(
                choreography_path
            ):
                with open(spotify_named_path, "w") as f:
                    json.dump(output_data, f, indent=2)
                print(f"✓ Also saved: {spotify_named_path}")

        print("\n✅ Analysis and choreography generation complete!")

    except Exception as e:
        print(f"❌ Choreography generation failed: {e}")
        print(
            "Music map was saved successfully - run choreography generation separately if needed"
        )
        sys.exit(1)
