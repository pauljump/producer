"""The musician's self-check: Muse Spark Open (open weights, on Baseten) as the session engineer.

Stage 3 finding (docs/SELF-CHECK-TEST.md): by ear alone, none of the Meta models reliably tells
a bass that is a semitone out or 150 ms late from a good one (Spark Open 1/6 and 3/6, Spark 1.2
3/6 and 4/6, Wonder 3/6 and 3/6; Wonder did catch sloppy timing 6/6). A real engineer does not
judge by ear alone either: they look at the meters. So Spark Open hears the passage and calls
measuring tools on the rendered audio (timing against the recording's own onsets, pitch against
the guitar, energy in the singer's band), then gives the verdict and a fix the player can act
on. The verdict and the instruction are the model's; the numbers are evidence.
"""
import base64
import json
import tempfile
from pathlib import Path

import librosa
import numpy as np

from . import hands, models
from .ears import parse_json

SYSTEM = ("You have native audio input; attached audio is audible to you. You are the session "
          "engineer. You check a new part before the producer hears it: is it in time, in tune, "
          "and leaving the singer room? Listen, then use the measuring tools, then decide. You do "
          "not judge taste; only whether the take is technically fit to play for the producer.")

ASK = """Passage {start:.1f}s to {end:.1f}s. Attached: the recording with the new {instrument} part.
Measure what you need (call the tools), then answer. JSON only:
{{"time": "together|drags|rushes|sloppy", "tune": "in tune|clashes", "room": "leaves room|crowds the singer",
 "level": "sits right|too loud|too quiet", "pass": true/false,
 "evidence": "which measurements and what you heard",
 "fix": "one plain instruction for the player if it fails, else empty"}}"""

TOOLS = [
    {"type": "function", "function": {"name": "measure_timing",
     "description": "Onset timing of the new part against the recording's own onsets in this passage: median offset in ms (positive = late) and spread in ms.",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "measure_tuning",
     "description": "How much of the new part's pitch energy falls on pitches the guitar is not playing, beat by beat (0 = all shared, 1 = none shared).",
     "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "measure_room",
     "description": "Energy of the new part in the singer's band (300 to 3500 Hz) relative to the vocal, and the part's overall loudness relative to the recording.",
     "parameters": {"type": "object", "properties": {}}}},
]

REFUSALS = ("don't have the ability", "cannot hear", "can't hear", "no audio", "not able to receive",
            "text-based", "unable to evaluate", "don't have access to audio", "cannot listen", "can't listen")
SR = 22050


def refused(text: str) -> bool:
    t = (text or "").lower()
    return any(r in t for r in REFUSALS)


def _mp3(audio: np.ndarray) -> dict:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.mp3"
        hands.save(p, audio)
        data = base64.b64encode(p.read_bytes()).decode()
    return {"type": "audio_url", "audio_url": {"url": f"data:audio/mpeg;base64,{data}"}}


def _mono(x: np.ndarray) -> np.ndarray:
    y = x.mean(axis=1) if x.ndim > 1 else x
    return librosa.resample(y.astype(np.float32), orig_sr=hands.SR, target_sr=SR)


class Meters:
    """Measurements on rendered audio for one passage."""

    def __init__(self, song, part, harmony, vocals, start, end, beats, chords=None):
        a, b = int(start * hands.SR), int(end * hands.SR)
        self.chords = [c for t, c in zip(beats, chords or []) if start <= t < end]
        self.song, self.part = _mono(song[a:b]), _mono(part[a:b])
        self.harmony = _mono(harmony[a:b]) if harmony is not None else self.song
        self.vocals = _mono(vocals[a:b]) if vocals is not None else None
        self.beats = [t - start for t in beats if start <= t < end]

    def measure_timing(self):
        hop = 64
        ref = librosa.onset.onset_detect(y=self.harmony, sr=SR, hop_length=hop, units="time", backtrack=False)
        new = librosa.onset.onset_detect(y=self.part, sr=SR, hop_length=hop, units="time", backtrack=False)
        if len(ref) < 3 or len(new) < 3:
            return {"note": "too few onsets in this passage to measure"}
        d = np.array([n - ref[np.argmin(np.abs(ref - n))] for n in new])
        d = d[np.abs(d) < 0.2] * 1000
        if len(d) < 3:
            return {"note": "new part's onsets do not line up with any recording onsets", "matched": 0}
        out = {"median_offset_ms": round(float(np.median(d)), 1),
               "spread_ms": round(float(np.percentile(d, 75) - np.percentile(d, 25)), 1),
               "matched_onsets": int(len(d)), "part_onsets": int(len(new))}
        # Baseline: how loosely the recording itself sits on its own beat grid (8th-note positions).
        if len(self.beats) > 2:
            b = np.asarray(self.beats)
            grid = np.sort(np.concatenate([b, (b[:-1] + b[1:]) / 2]))
            g = np.array([r - grid[np.argmin(np.abs(grid - r))] for r in ref]) * 1000
            g = g[np.abs(g) < 200]
            if len(g) > 3:
                out["recording_own_spread_ms"] = round(float(np.percentile(g, 75) - np.percentile(g, 25)), 1)
                out["note"] = ("the new part is placed on the recording's measured beats; a spread close to the "
                               "recording's own spread is the live performance breathing, not the new part")
        return out

    def measure_tuning(self):
        cp = librosa.feature.chroma_cqt(y=self.part, sr=SR)
        ch = librosa.feature.chroma_cqt(y=librosa.effects.harmonic(self.harmony), sr=SR)
        n = min(cp.shape[1], ch.shape[1])
        cp, ch = cp[:, :n], ch[:, :n]
        frames = librosa.time_to_frames(self.beats, sr=SR) if len(self.beats) > 2 else None
        if frames is not None:
            cp, ch = librosa.util.sync(cp, frames), librosa.util.sync(ch, frames)
        loud = cp.max(axis=0) > 0.3
        allowed = ch / (ch.max(axis=0, keepdims=True) + 1e-9) >= 0.4
        if self.chords and frames is not None:
            # A keys or bass part may play chord tones the guitar happens not to ring (a third
            # over a power chord). Count the measured chord's tones as shared, not foreign.
            from . import analysis
            names, T = analysis._chord_templates()
            for col in range(allowed.shape[1]):
                c = self.chords[min(max(col - 1, 0), len(self.chords) - 1)]
                if c in names:
                    allowed[:, col] |= T[names.index(c)] > 0
        foreign = ((cp / (cp.sum(axis=0, keepdims=True) + 1e-9)) * ~allowed).sum(axis=0)
        f = foreign[loud] if loud.any() else foreign
        return {"foreign_pitch_share_median": round(float(np.median(f)), 2),
                "beats_over_half_foreign": int((f > 0.5).sum()), "beats_measured": int(len(f)),
                "calibration": ("measured on this song: parts in key read 0.2 to 0.45 (overtones of sampled "
                                "instruments always add some); a part a semitone out reads 0.55 to 0.7")}

    def measure_room(self):
        def band(y, lo=300, hi=3500):
            s = np.abs(librosa.stft(y)) ** 2
            f = librosa.fft_frequencies(sr=SR)
            return float(s[(f >= lo) & (f < hi)].sum()) + 1e-12
        out = {"part_vs_recording_db": round(10 * np.log10(np.mean(self.part ** 2) / (np.mean(self.song ** 2) + 1e-12) + 1e-12), 1)}
        if self.vocals is not None:
            out["part_in_vocal_band_vs_vocal_db"] = round(10 * np.log10(band(self.part) / band(self.vocals)), 1)
        return out


def check(song, part, harmony, vocals, beats, instrument: str, start: float, end: float, label: str = "",
          tries: int = 3, chords: list[str] | None = None) -> dict:
    """One passage. `part` is the rendered part alone at its mix level; `harmony`/`vocals` are stems."""
    meters = Meters(song, part, harmony, vocals, start, end, beats, chords)
    drums = instrument == "drums"
    tools = [t for t in TOOLS if not (drums and t["function"]["name"] == "measure_tuning")]
    prompt = ASK.format(start=start, end=end, instrument=instrument)
    if drums:
        prompt = prompt.replace('"tune": "in tune|clashes"', '"tune": "n/a"') + (
            "\nDrums are unpitched: do not judge tuning. Judge time, room and level only.")
    mixed = song + part
    a, b = int(start * hands.SR), int(end * hands.SR)
    for _ in range(tries):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": [
            {"type": "text", "text": prompt}, _mp3(mixed[a:b])]}]
        measured = {}
        for _turn in range(5):
            resp = models.chat("baseten", models.SELF_CHECK, msgs, purpose=f"self-check {instrument} {label}".strip(),
                               estimate_usd=0.02, max_tokens=6000, tools=tools)
            m = resp.choices[0].message
            if not m.tool_calls:
                break
            msgs.append(m.model_dump(exclude_none=True))
            for call in m.tool_calls:
                fn = getattr(meters, call.function.name, None)
                r = fn() if fn else {"error": "unknown tool"}
                measured[call.function.name] = r
                msgs.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(r, default=float)})
        text = models.text(resp)
        if refused(text) and not measured:
            continue
        try:
            v = parse_json(text)
        except ValueError:
            continue
        v.update({"start": start, "end": end, "measured": measured})
        return v
    return {"start": start, "end": end, "pass": None, "error": "no verdict"}
