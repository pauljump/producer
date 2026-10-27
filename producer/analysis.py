"""Measured truth about a recording: beats, downbeats, tempo, key, chords, sections, energy.

This is plain signal processing. It makes no musical decisions; it gives the musicians'
ears (Muse Spark 1.2) facts to reconcile against, and the renderer a timeline to play on.
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np

SR = 22050
HOP = 512
PITCHES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Krumhansl-Kessler key profiles.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def beats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Beat and downbeat times in seconds, from beat_this (a neural beat tracker).

    librosa's tracker doubles slow ballads (it called a 91 BPM song 123), so it is not used here.
    """
    from beat_this.inference import File2Beats

    tracker = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    b, d = tracker(str(path))
    return np.asarray(b), np.asarray(d)


def estimate_key(chroma: np.ndarray) -> dict:
    profile = chroma.mean(axis=1)
    scores = []
    for tonic in range(12):
        for mode, ref in (("major", _MAJOR), ("minor", _MINOR)):
            r = np.corrcoef(profile, np.roll(ref, tonic))[0, 1]
            scores.append((r, tonic, mode))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    return {"tonic": PITCHES[best[1]], "mode": best[2], "confidence": round(float(best[0]), 3),
            "runner_up": f"{PITCHES[second[1]]} {second[2]}", "runner_up_confidence": round(float(second[0]), 3)}


def _chord_templates() -> tuple[list[str], np.ndarray]:
    names, rows = [], []
    shapes = {"": (0, 4, 7), "m": (0, 3, 7), "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10),
              "sus4": (0, 5, 7), "sus2": (0, 2, 7)}
    for root in range(12):
        for q, ivs in shapes.items():
            v = np.zeros(12)
            for i in ivs:
                v[(root + i) % 12] = 1.0
            v[root] += 0.5  # roots are usually loudest
            names.append(PITCHES[root] + q)
            rows.append(v / np.linalg.norm(v))
    names.append("N")
    rows.append(np.ones(12) / np.sqrt(12))
    return names, np.array(rows)


def chords(chroma_beats: np.ndarray, switch_cost: float = 2.5, sharp: float = 25.0) -> list[str]:
    """Best chord per beat, smoothed with a Viterbi pass that prefers to stay on a chord."""
    names, T = _chord_templates()
    c = chroma_beats / (np.linalg.norm(chroma_beats, axis=0, keepdims=True) + 1e-9)
    like = T @ c  # (chords, beats) cosine similarity
    like[-1] -= 1.0  # never "no chord"; quiet beats are handled by energy, not here
    like[[i for i, nm in enumerate(names) if "sus" in nm]] -= 0.04  # guitar voicings fool sus templates
    logp = like * sharp
    n = len(names)
    trans = np.full((n, n), -switch_cost)
    np.fill_diagonal(trans, 0.0)
    score = logp[:, 0].copy()
    back = np.zeros(logp.shape, dtype=int)
    for t in range(1, logp.shape[1]):
        cand = score[:, None] + trans
        back[:, t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + logp[:, t]
    path = [int(score.argmax())]
    for t in range(logp.shape[1] - 1, 0, -1):
        path.append(back[path[-1], t])
    return [names[i] for i in reversed(path)]


def sections(y: np.ndarray, downbeats: np.ndarray, k: int | None = None) -> list[dict]:
    """Candidate section boundaries on downbeats, from bar-level self-similarity."""
    if len(downbeats) < 4:
        return [{"start": 0.0, "end": float(len(y) / SR), "label": "A"}]
    frames = librosa.time_to_frames(downbeats, sr=SR, hop_length=HOP)
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP, n_mfcc=13)
    feat = np.vstack([librosa.util.normalize(chroma, axis=0), librosa.util.normalize(mfcc, axis=1)])
    bars = librosa.util.sync(feat, frames, aggregate=np.median)
    nbars = bars.shape[1]
    k = k or int(np.clip(round(nbars / 8), 2, 8))
    bounds = librosa.segment.agglomerative(bars, k)
    bounds = sorted(set(int(b) for b in bounds) | {0})
    times = list(downbeats[[min(b, len(downbeats) - 1) for b in bounds]]) + [len(y) / SR]
    # Label repeats: segments whose mean features are close get the same letter.
    means = [bars[:, bounds[i]:(bounds[i + 1] if i + 1 < len(bounds) else nbars)].mean(axis=1)
             for i in range(len(bounds))]
    labels, protos = [], []
    for m in means:
        for j, p in enumerate(protos):
            if np.dot(m, p) / (np.linalg.norm(m) * np.linalg.norm(p) + 1e-9) > 0.97:
                labels.append(chr(65 + j))
                break
        else:
            protos.append(m)
            labels.append(chr(65 + len(protos) - 1))
    return [{"start": round(float(times[i]), 3), "end": round(float(times[i + 1]), 3), "label": labels[i]}
            for i in range(len(bounds))]


def load_mix(paths: list[Path]) -> np.ndarray:
    ys = [librosa.load(str(p), sr=SR, mono=True)[0] for p in paths]
    n = min(len(y) for y in ys)
    return sum(y[:n] for y in ys)


def analyze(path: Path, harmony: list[Path] | None = None) -> dict:
    """harmony: stems that carry the chords (e.g. separated guitar); falls back to the full mix."""
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    duration = len(y) / SR
    b, d = beats(path)
    ibi = np.diff(b)
    tempo = 60.0 / float(np.median(ibi)) if len(ibi) else 0.0
    beats_per_bar = int(round(np.median(np.diff(np.searchsorted(b, d))))) if len(d) > 2 else 4

    harmonic = librosa.effects.harmonic(load_mix(harmony) if harmony else y)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=SR, hop_length=HOP)
    key = estimate_key(chroma)
    bf = librosa.time_to_frames(b, sr=SR, hop_length=HOP)
    chroma_b = librosa.util.sync(chroma, bf, aggregate=np.median)[:, 1:]  # column i = beat i to beat i+1
    chord_seq = chords(chroma_b)[: len(b)]

    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    df = librosa.time_to_frames(d, sr=SR, hop_length=HOP)
    bar_rms = librosa.util.sync(rms[None, :], df, aggregate=np.mean)[0][1:]
    bar_db = librosa.amplitude_to_db(bar_rms + 1e-9, ref=np.max(bar_rms) + 1e-9)

    return {
        "file": path.name,
        "duration": round(duration, 3),
        "tempo_bpm": round(tempo, 2),
        "tempo_stability": round(float(ibi.std() / ibi.mean()), 4) if len(ibi) else None,
        "beats_per_bar": beats_per_bar,
        "beats": [round(float(t), 3) for t in b],
        "downbeats": [round(float(t), 3) for t in d],
        "key": key,
        "chords_per_beat": chord_seq,
        "chroma_per_beat": [[round(float(v), 3) for v in col] for col in chroma_b.T],
        "harmony_source": [p.name for p in harmony] if harmony else "full mix",
        "chord_changes": _changes(b, chord_seq),
        "bar_energy_db": [round(float(x), 1) for x in bar_db],
        "sections": sections(y, d),
        "method": "beat_this final0 (beats); Krumhansl-Kessler (key); chord templates + Viterbi on "
                  "harmonic CQT chroma (chords); bar self-similarity agglomerative (sections)",
    }


def _changes(b: np.ndarray, seq: list[str]) -> list[dict]:
    out = []
    for t, c in zip(b, seq):
        if not out or out[-1]["chord"] != c:
            out.append({"time": round(float(t), 3), "chord": c})
    return out


if __name__ == "__main__":
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".analysis.json")
    stems = [Path(p) for p in sys.argv[3:]] or None
    a = analyze(src, stems)
    out.write_text(json.dumps(a, indent=1))
    print(json.dumps({k: a[k] for k in ("duration", "tempo_bpm", "tempo_stability", "beats_per_bar", "key")}))
    print("sections", [(s["label"], s["start"]) for s in a["sections"]])
    print("chords", [c["chord"] for c in a["chord_changes"]][:40])
