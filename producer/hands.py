"""The musicians' hands: play written notes with sampled instruments on the song's own timeline.

A part is a list of notes the musician wrote, positioned in *beats* of the recording
(beat 0 = the first tracked beat). The renderer maps beats to seconds through the measured
beat grid, so a part follows a live performance that speeds up and slows down. Nothing here
chooses a note; it only plays what it is given.

Part format:
{"instrument": "bass" | "drums" | "keys" | ...,
 "notes": [{"beat": 12.0, "len": 0.9, "pitch": 45, "vel": 90}, ...]}
pitch is MIDI (drums use General MIDI drum numbers, e.g. 36 kick, 38 snare, 42 closed hat).
"""
import subprocess
import tempfile
from pathlib import Path

import mido
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

SR = 44100
SOUNDFONT = Path.home() / ".cache" / "producer" / "GeneralUser-GS.sf2"

# General MIDI programs for the session instruments.
PROGRAMS = {"bass": 33, "upright bass": 32, "fretless bass": 35, "keys": 4, "piano": 0, "organ": 16,
            "strings": 48, "pad": 89, "electric guitar": 27, "nylon guitar": 24, "cello": 42}


def beat_to_seconds(beats: list[float], b: float) -> float:
    """Map a fractional beat position to seconds on the measured grid (linear between beats,
    extrapolated at the ends with the nearest beat interval)."""
    grid = np.asarray(beats)
    if b <= 0:
        return float(grid[0] + b * (grid[1] - grid[0]))
    if b >= len(grid) - 1:
        return float(grid[-1] + (b - (len(grid) - 1)) * (grid[-1] - grid[-2]))
    i = int(np.floor(b))
    return float(grid[i] + (b - i) * (grid[i + 1] - grid[i]))


def to_midi(part: dict, beats: list[float], path: Path) -> None:
    """Write a MIDI file whose ticks are absolute seconds on the song's timeline."""
    tpb, tempo = 1000, 1_000_000  # 1 beat = 1 s at 60 BPM; ticks are milliseconds
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    drums = part["instrument"] == "drums"
    ch = 9 if drums else 0
    if not drums:
        track.append(mido.Message("program_change", program=PROGRAMS.get(part["instrument"], 33), channel=ch, time=0))
    events = []
    for n in part["notes"]:
        on = beat_to_seconds(beats, n["beat"])
        off = beat_to_seconds(beats, n["beat"] + max(n.get("len", 0.5), 0.05))
        vel = int(np.clip(n.get("vel", 90), 1, 127))
        events.append((on, 1, mido.Message("note_on", note=int(n["pitch"]), velocity=vel, channel=ch)))
        events.append((off, 0, mido.Message("note_off", note=int(n["pitch"]), velocity=0, channel=ch)))
    events.sort(key=lambda e: (e[0], e[1]))
    now = 0
    for t, _, msg in events:
        tick = int(round(max(t, 0) * 1000))
        msg.time = tick - now
        now = tick
        track.append(msg)
    mid.save(str(path))


def render(part: dict, beats: list[float], duration: float) -> np.ndarray:
    """Render a part to a stereo float array exactly `duration` seconds long."""
    with tempfile.TemporaryDirectory() as d:
        mid, wav = Path(d) / "part.mid", Path(d) / "part.wav"
        to_midi(part, beats, mid)
        subprocess.run(["fluidsynth", "-ni", "-q", "-g", "0.8", "-r", str(SR), "-F", str(wav),
                        str(SOUNDFONT), str(mid)], check=True, capture_output=True)
        audio, sr = sf.read(str(wav), always_2d=True)
    n = int(round(duration * SR))
    out = np.zeros((n, 2))
    out[: min(n, len(audio))] = audio[:n]
    return out


def loudness(x: np.ndarray) -> float:
    return pyln.Meter(SR).integrated_loudness(x) if np.abs(x).max() > 1e-6 else -70.0


def at_level(song: np.ndarray, audio: np.ndarray, rel: float) -> np.ndarray:
    """Scale a rendered part so it sits `rel` LU relative to the song's loudness."""
    lu = loudness(audio)
    return audio * 10 ** ((loudness(song) + rel - lu) / 20) if lu > -70 else audio


def mix(song: np.ndarray, parts: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """Mix rendered parts under the song. Each part comes with a level in LU relative to the
    song's loudness (e.g. -4.0 = four LU under the recording); levels are the musicians' call."""
    out = song.copy()
    for audio, rel in parts:
        out += at_level(song, audio, rel)
    peak = np.abs(out).max()
    return out / peak * 0.95 if peak > 0.95 else out


def load(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), always_2d=True)
    if sr != SR:
        raise ValueError(f"{path} is {sr} Hz; convert to {SR}")
    return audio if audio.shape[1] == 2 else np.repeat(audio, 2, axis=1)


def save(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".mp3":
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, audio, SR)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f.name, "-b:a", "192k", str(path)], check=True)
    else:
        sf.write(str(path), audio, SR)
