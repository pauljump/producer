"""Hum it or beatbox it: turn a few seconds of the producer's voice into a musical sketch.

Measuring only. Pitch tracking (pYIN) turns a hum into notes; onset detection plus a simple
spectral split turns a beatbox into kick / snare / hat hits. Times are converted to beats at
the song's tempo. Muse Spark 1.2 listens to the same clip and says what the producer seems to
want from it (the feel), and the musician decides how to play the idea in character. Our code
never decides where or how the idea is used.
"""
import base64
import subprocess
from pathlib import Path

import librosa
import numpy as np

from . import models
from .ears import parse_json

SR = 22050


def _convert(raw: bytes, *args: str) -> bytes:
    """Voice memos (m4a) keep their index at the end, so ffmpeg needs a real file, not a pipe."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        src, out = Path(d) / "in", Path(d) / "out.wav"
        src.write_bytes(raw)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), *args, str(out)], check=True)
        return out.read_bytes()


def load(raw: bytes) -> np.ndarray:
    wav = _convert(raw, "-ac", "1", "-ar", str(SR))
    y, _ = librosa.load(__import__("io").BytesIO(wav), sr=SR, mono=True)
    return np.concatenate([np.zeros(int(0.25 * SR)), y])  # onsets at the very first frame are otherwise lost


def hum_to_notes(y: np.ndarray, bpm: float) -> list[dict]:
    """Notes the producer hummed: {beat, len (beats), note, midi}, starting at beat 0."""
    hop = 256
    f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=900, sr=SR, hop_length=hop, frame_length=2048)
    midi = np.where(voiced, librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0)), np.nan)
    # A real voice wobbles and slides; smooth the pitch over ~0.1 s so one sung note stays one note.
    import scipy.ndimage as ndi
    filled = np.where(np.isnan(midi), 0, midi)
    smooth = ndi.median_filter(filled, size=9)
    midi = np.where(np.isnan(midi), np.nan, smooth)
    t = librosa.frames_to_time(np.arange(len(midi)), sr=SR, hop_length=hop)
    notes, cur = [], None
    for i, m in enumerate(midi):
        r = None if np.isnan(m) else int(round(m))
        if cur and (r is None or abs(m - np.median(cur["vals"])) >= 0.75):
            notes.append(cur)
            cur = None
        if r is not None and cur is None:
            cur = {"start": t[i], "end": t[i], "midi": r, "vals": []}
        if cur is not None:
            cur["end"] = t[i] + hop / SR
            cur["vals"].append(m)
    if cur:
        notes.append(cur)
    notes = [n for n in notes if n["end"] - n["start"] >= 0.12]
    if notes:  # drop stray tracks far below the hummed line (breath, room rumble)
        med = np.median([np.median(n["vals"]) for n in notes])
        notes = [n for n in notes if abs(np.median(n["vals"]) - med) < 15]
    if not notes:
        return []
    t0, spb = notes[0]["start"], 60.0 / bpm
    out = []
    for n in notes:
        m = int(round(np.median(n["vals"])))
        out.append({"beat": round((n["start"] - t0) / spb * 4) / 4, "len": max(0.25, round((n["end"] - n["start"]) / spb * 4) / 4),
                    "note": librosa.midi_to_note(m, unicode=False), "midi": m})
    return out


def beatbox_to_hits(y: np.ndarray, bpm: float) -> list[dict]:
    """Hits the producer beatboxed: {beat, hit: kick|snare|hat}, starting at beat 0."""
    hop = 256
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)
    onsets = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=hop, units="samples",
                                        backtrack=False, wait=int(0.09 * SR / hop), delta=0.2)
    if len(onsets) == 0:
        return []
    spb, t0 = 60.0 / bpm, onsets[0] / SR
    hits = []
    for o in onsets:
        seg = y[max(0, o - 256):o + int(0.08 * SR)]
        if len(seg) < 512:
            continue
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        f = np.fft.rfftfreq(len(seg), 1 / SR)
        tot = spec.sum() + 1e-12
        low, high = spec[f < 250].sum() / tot, spec[f > 5000].sum() / tot
        hit = "kick" if low > 0.4 else ("hat" if high > 0.4 else "snare")
        hits.append({"beat": round((o / SR - t0) / spb * 4) / 4, "hit": hit})
    return hits


FEEL = """The producer of a recording session just {verb} this short idea into their phone for the
{instrument} player. Listen. They are not a musician. What are they going for?
JSON only: {{"what_it_is": "a melody / a riff / a rhythm / a groove ...",
 "feel": "how it should feel when played (e.g. lazy, pushing, sad, bouncy)",
 "words_if_any": "anything they said, or empty"}}"""


def hear_sketch(raw: bytes, kind: str, instrument: str) -> dict:
    wav = _convert(raw, "-ac", "1", "-ar", "16000")
    resp = models.chat("meta", models.EARS, [
        {"role": "system", "content": "You have native audio input; the attached audio is audible to you."},
        {"role": "user", "content": [{"type": "text", "text": FEEL.format(verb="hummed" if kind == "hum" else "beatboxed", instrument=instrument)},
                                     {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}}]}],
        purpose=f"ears: producer {kind} sketch", estimate_usd=0.02, max_tokens=3000)
    try:
        return parse_json(models.text(resp))
    except ValueError:
        return {}


def sketch(raw: bytes, kind: str, bpm: float, instrument: str) -> dict:
    y = load(raw)
    measured = hum_to_notes(y, bpm) if kind == "hum" else beatbox_to_hits(y, bpm)
    return {"kind": kind, "measured": measured, "heard": hear_sketch(raw, kind, instrument),
            "seconds": round(len(y) / SR, 1)}
