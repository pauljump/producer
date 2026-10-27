"""The song's chart, built from measurements, and a part's measured fit against the recording.

These are facts handed to the musicians as evidence. They decide nothing: the musician reads
the chart, writes notes, and reads the fit report like a session player reading a chart and
listening back.
"""
from collections import Counter

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def note_to_midi(name: str) -> int:
    """'A1' -> 33, 'F#2' -> 42, 'Bb1' -> 34."""
    name = name.strip()
    i = 2 if len(name) > 1 and name[1] in "#b" else 1
    pc, octave = name[:i], int(name[i:])
    pc = FLAT.get(pc, pc)
    return NOTE_NAMES.index(pc) + 12 * (octave + 1)


def midi_to_note(m: int) -> str:
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def bar_starts(a: dict) -> list[int]:
    """Index into a['beats'] of each downbeat."""
    beats = np.asarray(a["beats"])
    return [int(np.argmin(np.abs(beats - d))) for d in a["downbeats"]]


def chart(a: dict, song_map: dict | None = None, vocal_rms: list[float] | None = None) -> list[dict]:
    """One row per bar: where it is, what the guitar is playing (measured), how loud, which section."""
    starts = bar_starts(a)
    beats = a["beats"]
    rows = []
    for n, s in enumerate(starts):
        e = starts[n + 1] if n + 1 < len(starts) else len(beats)
        chroma = np.mean(a["chroma_per_beat"][s:e], axis=0) if e > s else np.zeros(12)
        top = [NOTE_NAMES[i] for i in np.argsort(chroma)[::-1][:4]]
        per_beat = a.get("chords_in_key_per_beat", a["chords_per_beat"])[s:e]
        t = beats[s]
        row = {"bar": n + 1, "beat": s, "beats_in_bar": e - s, "time": round(t, 2),
               "chords_by_beat": per_beat, "strongest_pitches": top,
               "energy_db": a["bar_energy_db"][n] if n < len(a["bar_energy_db"]) else None}
        if song_map:
            for sec in song_map.get("sections", []):
                if sec["start"] - 0.3 <= t < sec["end"]:
                    row["section"] = sec["label"]
        if vocal_rms is not None and n < len(vocal_rms):
            row["singing"] = bool(vocal_rms[n] > 0.02)
        rows.append(row)
    return rows


def to_beats(a: dict, written: list[dict]) -> list[dict]:
    """Convert a musician's notes ({bar, beat (1-based within bar), len, note, vel}) to absolute
    beat positions on the recording's grid."""
    starts = bar_starts(a)
    out = []
    for n in written:
        bar = int(n["bar"]) - 1
        if not 0 <= bar < len(starts):
            continue
        pitch = n["pitch"] if "pitch" in n else note_to_midi(n["note"])
        out.append({"beat": starts[bar] + float(n["beat"]) - 1.0, "len": float(n.get("len", 1.0)),
                    "pitch": int(pitch), "vel": int(n.get("vel", 90))})
    return out


def fit(a: dict, notes: list[dict], drums: bool = False) -> dict:
    """Measured fit of a part against the recording: are pitches supported by what the guitar
    is playing on that beat, where do notes land against the grid, how busy is each bar."""
    chroma = np.asarray(a["chroma_per_beat"])
    starts = bar_starts(a)
    supported, unsupported, offgrid = 0, [], 0
    per_bar = Counter()
    for n in notes:
        b = n["beat"]
        bar = int(np.searchsorted(starts, b, side="right"))
        per_bar[bar] += 1
        frac = b - np.floor(b)
        if min(abs(frac - q) for q in (0, 0.25, 0.5, 0.75, 1.0)) > 0.06:
            offgrid += 1
        if drums:
            continue
        i = int(np.clip(np.floor(b), 0, len(chroma) - 1))
        c = chroma[i] / (chroma[i].max() + 1e-9)
        if c[n["pitch"] % 12] >= 0.45:
            supported += 1
        else:
            unsupported.append({"bar": bar, "beat_in_bar": round(b - starts[bar - 1] + 1, 2) if bar else None,
                                "note": midi_to_note(n["pitch"]),
                                "guitar_strongest": [NOTE_NAMES[j] for j in np.argsort(c)[::-1][:3]]})
    total = len(notes) or 1
    report = {"notes": len(notes), "off_grid_notes": offgrid,
              "busiest_bars": per_bar.most_common(3), "notes_per_bar_avg": round(len(notes) / max(len(per_bar), 1), 2)}
    if not drums:
        pitches = [n["pitch"] for n in notes] or [0]
        report.update({"pitch_supported_by_guitar": round(supported / total, 2),
                       "unsupported_notes": unsupported[:12],
                       "range": f"{midi_to_note(min(pitches))} to {midi_to_note(max(pitches))}"})
    return report
