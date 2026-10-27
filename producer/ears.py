"""The musicians' ears: Meta models listen to the recording, phrase by phrase, then as a whole.

Stage-1 finding (see docs/LISTENING-TEST.md): Muse Spark hears words and broad character
reliably, but a leading question ("describe the band") makes it invent instruments. So the
ears ask narrow, falsifiable questions about short, bar-aligned phrases, where timestamps are
true by construction, and the bandleader (Muse Wonder) hears the whole song with those notes
and the measurements, and writes the song map the musicians work from.
"""
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import models

SYSTEM = ("You have native audio input; the attached audio is audible to you. Answer only from "
          "what you hear. If you are not sure, say so. Never assume instruments that a typical "
          "band would have.")

PHRASE = """This is one phrase ({start:.1f}s to {end:.1f}s) of a recording. Listen closely.
JSON only:
{{"singing": true/false,
 "words": "the sung or spoken words you can make out, or empty",
 "instruments_certain": ["only instruments you can clearly hear"],
 "playing": "one short phrase on how the instruments are played here (e.g. steady strumming, picking, stops)",
 "intensity": 1-5,
 "notable": "anything that stands out (a stop, a hit, a held note, a laugh, crowd), or empty"}}"""

SONG_MAP = """You are the bandleader preparing a recording session. You have heard the whole
recording (attached). Below are (a) phrase-by-phrase listening notes from your ears, with true
timestamps, and (b) measurements from signal analysis (beats, tempo, key, chords). Measurements
can be wrong in known ways (tempo doubled or halved, sus chords mistaken for triads, rough
section boundaries). Your ears can be wrong too. Reconcile them and say where you sided with
which.

PHRASE NOTES:
{phrases}

MEASURED:
{measured}

Write the song map the session musicians will play from. JSON only:
{{"title_guess": "...",
 "who_is_on_the_recording": ["the performers and instruments really on it"],
 "tempo_bpm": n, "meter": "...", "key": "...",
 "reconciliation": "where measurements and ears disagreed and what you decided",
 "sections": [{{"start": s, "end": s, "label": "intro|verse 1|chorus 1|...", "lyrics": "...", "energy": 1-5,
               "chords": "progression", "what_it_feels_like": "...", "room_for_new_parts": "..."}}],
 "the_song_is_about": "one sentence, from the words",
 "feel": "how it moves and breathes, for players",
 "do_not": ["how new players (bass, drums, keys and so on will be added) must not play on this recording"]}}"""


def cut(src: Path, start: float, end: float, fmt: str = "wav") -> bytes:
    with tempfile.NamedTemporaryFile(suffix="." + fmt) as f:
        args = ["-ar", "16000", "-ac", "1"] if fmt == "wav" else ["-ac", "1", "-b:a", "64k"]
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(src),
                        "-t", f"{end - start:.3f}", *args, f.name], check=True)
        return Path(f.name).read_bytes()


def parse_json(s: str) -> dict:
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        raise ValueError(f"no JSON in reply: {s[:300]}")
    return json.loads(m.group(0))


def phrases(analysis: dict, bars: int = 4) -> list[tuple[float, float]]:
    d = analysis["downbeats"]
    edges = [0.0] + d[bars::bars] + [analysis["duration"]]
    if len(d) > bars:
        edges[0] = 0.0
    out = []
    for a, b in zip(edges, edges[1:]):
        if b - a > 1.0:
            out.append((a, b))
    return out


def _audio(b: bytes, fmt: str = "wav") -> dict:
    import base64
    return {"type": "input_audio", "input_audio": {"data": base64.b64encode(b).decode(), "format": fmt}}


def hear_phrase(src: Path, start: float, end: float, model: str = models.EARS) -> dict:
    resp = models.chat("meta", model, [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [{"type": "text", "text": PHRASE.format(start=start, end=end)},
                                     _audio(cut(src, start, end))]}],
        purpose=f"ears phrase: {src.stem}", estimate_usd=0.02, max_tokens=3000)
    try:
        note = parse_json(models.text(resp))
    except ValueError:
        note = {"unparsed": models.text(resp)[:500]}
    return {"start": round(start, 2), "end": round(end, 2), **note}


def measured_summary(a: dict) -> dict:
    return {"duration_s": a["duration"], "tempo_bpm": a["tempo_bpm"], "beats_per_bar": a["beats_per_bar"],
            "key": f'{a["key"]["tonic"]} {a["key"]["mode"]} (runner-up {a["key"]["runner_up"]})',
            "chord_changes": [(c["time"], c["chord"]) for c in a["chord_changes"]],
            "bar_energy_db": a["bar_energy_db"], "downbeats_s": a["downbeats"]}


def song_map(src: Path, analysis: dict, notes: list[dict]) -> dict:
    whole = cut(src, 0, analysis["duration"])
    resp = models.chat("meta", models.WONDER, [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": SONG_MAP.format(phrases=json.dumps(notes), measured=json.dumps(measured_summary(analysis)))},
            _audio(whole)]}],
        purpose=f"bandleader song map: {src.stem}", estimate_usd=0.25, max_tokens=12000)
    return parse_json(models.text(resp))


def listen(src: Path, analysis: dict) -> dict:
    with ThreadPoolExecutor(6) as pool:
        notes = list(pool.map(lambda p: hear_phrase(src, *p), phrases(analysis)))
    return {"phrases": notes, "song_map": song_map(src, analysis, notes)}


if __name__ == "__main__":
    wav = Path(sys.argv[1])
    analysis = json.loads(wav.with_suffix(".analysis.json").read_text())
    result = listen(wav, analysis)
    wav.with_suffix(".ears.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result["song_map"], indent=1))
