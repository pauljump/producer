"""The band: Muse Wonder as bandleader and as each hired musician.

The bandleader casts a musician (a character: feel, touch, register, signature move,
influences) for the role the producer asked for. The musician is a separate Wonder agent with
that character as its system prompt. It hears the recording, reads the chart and the song map,
writes its part as notes, and uses tools to check the measured fit before handing the take in.
Every note in a take was written by the model.
"""
import json
from pathlib import Path

from . import models, score
from .ears import parse_json

CAST = """You are the bandleader on a recording session. The producer (who knows nothing about
music, only what they like) just said: "{ask}".

SONG MAP (what we heard):
{song_map}

Cast the musician. Give them a real character, so their playing has a point of view that fits
this song. JSON only:
{{"name": "a first name", "instrument": "one of: bass, upright bass, fretless bass, drums, keys, piano, organ, strings, pad, electric guitar, cello",
 "feel": "...", "touch": "...", "register": "...", "signature_move": "...",
 "influences": ["2-3 real players"], "one_line": "how they'd introduce themselves to the producer",
 "plan": "what they will play and where, in one or two plain sentences the producer can understand"}}"""

PLAYER = """You are {name}, a session {instrument} player hired for this recording.
Feel: {feel}. Touch: {touch}. Register: {register}. Signature move: {signature_move}.
Influences: {influences}.

You play over a real recording that already exists. You never change it; you add one part that
sits in time and in key under it, and leaves room for the singer. You write every note yourself.

How you work:
1. Listen to the recording (attached) and read the chart: one row per bar, with the beats in
   that bar, the chords measured on each beat, the pitches the guitar is ringing loudest, the
   loudness, and the section.
2. Write your part as notes: {{"bar": n, "beat": 1-based position in the bar (1, 1.5, 2.75 ...),
   "len": length in beats, "note": "A1" style name{drum_hint}, "vel": 1-127}}.
3. Call check_fit with your notes. It reports what the measurements say: which of your
   pitches the guitar is not playing on that beat, notes off the grid, how busy each bar is.
   A clash can be a deliberate passing note; decide like a player would. But if your
   unsupported notes cluster on the same beats bar after bar, your harmonic rhythm is wrong:
   chords often change mid-bar, so read chords_by_beat, not the song map's summary. A part that
   sits well usually has 85% or more of its notes supported.
4. Revise if you need to, then call submit_take with the final notes and a short note to the
   producer in plain words (no theory) about what you played."""

TOOLS = [
    {"type": "function", "function": {
        "name": "check_fit",
        "description": "Measure how the written notes sit against the recording (pitch support from the guitar on each beat, grid alignment, density).",
        "parameters": {"type": "object", "properties": {"notes": {"type": "array", "items": {"type": "object"}}},
                       "required": ["notes"]}}},
    {"type": "function", "function": {
        "name": "submit_take",
        "description": "Hand in the final take.",
        "parameters": {"type": "object", "properties": {
            "notes": {"type": "array", "items": {"type": "object"}},
            "level_lu": {"type": "number", "description": "how loud your part sits relative to the recording, in LU (e.g. -6 = well under it)"},
            "to_producer": {"type": "string"}},
            "required": ["notes", "level_lu", "to_producer"]}}},
]

GM_DRUMS = ' (drums: use "kick", "snare", "rim", "closed hat", "open hat", "pedal hat", "ride", "crash", "low tom", "mid tom", "high tom", "shaker", "tambourine")'
DRUM_MAP = {"kick": 36, "snare": 38, "rim": 37, "closed hat": 42, "open hat": 46, "pedal hat": 44, "ride": 51,
            "crash": 49, "low tom": 45, "mid tom": 47, "high tom": 50, "shaker": 70, "tambourine": 54}


TRANSLATE = """You are the bandleader. The producer knows nothing about music, only what they like,
and they just said this about {name}'s {instrument} take: "{words}"

Translate their taste into a musical instruction a session player can act on. Read their words
generously: "I don't believe it" usually means the part sounds generic or effortful, not that a
note is wrong. "that, but less" means keep the idea and reduce. Be specific (density, register,
rhythm, dynamics, where in the song), and keep what they did not complain about.

SONG MAP: {song_map}
THE TAKE THEY HEARD (notes): {take}
WHAT THE PLAYER TOLD THEM: {said}

JSON only:
{{"heard_as": "what you think the producer means, in one sentence",
 "instruction": "the direction to the player, musical and specific",
 "where": "the whole song, or which sections/bars",
 "keep": "what must not change",
 "reply_to_producer": "one short plain sentence back to the producer (no theory)"}}"""


def translate(words: str, musician: dict, take: dict, song_map: dict) -> dict:
    """Taste in, musical instruction out (Muse Wonder as bandleader)."""
    resp = models.chat("meta", models.WONDER, [{"role": "user", "content": TRANSLATE.format(
        words=words, name=musician["name"], instrument=musician["instrument"], song_map=json.dumps(song_map),
        take=json.dumps(take["written"]), said=take.get("to_producer", ""))}],
        purpose="bandleader: translate taste", estimate_usd=0.15, max_tokens=8000)
    return parse_json(models.text(resp))


ORIGINALS = """You are the bandleader. Before anyone new is hired, introduce the people already on
this recording as members of the band. Give each one a character, as if they had made this
music on purpose: name, feel, what they did on this take. They are the heart of the session.

SONG MAP: {song_map}

JSON only: {{"originals": [{{"name": "a first name", "role": "e.g. singer and guitarist",
 "feel": "...", "what_they_played": "one plain sentence about this recording",
 "one_line": "how they'd introduce themselves to the producer"}}]}}
One entry per real performer (not per instrument if one person plays two)."""


def cast_originals(song_map: dict) -> list[dict]:
    resp = models.chat("meta", models.WONDER, [{"role": "user", "content": ORIGINALS.format(song_map=json.dumps(song_map))}],
                       purpose="bandleader: characters for the original performers", estimate_usd=0.1, max_tokens=12000,
                       reasoning_effort="low")
    return parse_json(models.text(resp)).get("originals", [])


def cast(ask: str, song_map: dict) -> dict:
    resp = models.chat("meta", models.WONDER, [{"role": "user", "content": CAST.format(ask=ask, song_map=json.dumps(song_map))}],
                       purpose="bandleader: cast musician", estimate_usd=0.1, max_tokens=6000)
    return parse_json(models.text(resp))


def _norm(notes: list[dict], drums: bool) -> list[dict]:
    out = []
    for n in notes:
        n = dict(n)
        if drums:
            name = str(n.get("note", "")).lower()
            if name not in DRUM_MAP:
                continue
            n["pitch"] = DRUM_MAP[name]
        out.append(n)
    return out


def play(musician: dict, analysis: dict, song_map: dict, chart: list[dict], audio_part: dict,
         direction: str | None = None, previous: dict | None = None, max_turns: int = 8,
         engineer=None, revisions: int = 2) -> dict:
    """One take. Returns {'notes' (absolute beats), 'written', 'level_lu', 'to_producer', 'checks',
    'engineer'}. If `engineer(take) -> (ok, verdicts)` is given, a submitted take goes to the
    self-check first; a failed check goes back to the musician to revise, up to `revisions` times."""
    drums = musician["instrument"] == "drums"
    system = PLAYER.format(drum_hint=GM_DRUMS if drums else "", **{k: musician.get(k, "") for k in
                           ("name", "instrument", "feel", "touch", "register", "signature_move", "influences")})
    brief = {"song_map": song_map, "chart": chart, "your_plan": musician.get("plan")}
    if previous:
        brief["your_last_take"] = previous["written"]
        brief["direction"] = direction
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": json.dumps(brief)}, audio_part]}]
    checks, reviews = [], []
    for _ in range(max_turns):
        resp = models.chat("meta", models.WONDER, msgs, purpose=f"musician take: {musician['name']} ({musician['instrument']})",
                           estimate_usd=0.4, max_tokens=16000, tools=TOOLS)
        m = resp.choices[0].message
        msgs.append(m.model_dump(exclude_none=True))
        if not m.tool_calls:
            msgs.append({"role": "user", "content": "Call check_fit, then submit_take."})
            continue
        for call in m.tool_calls:
            args = json.loads(call.function.arguments)
            written = _norm(args.get("notes", []), drums)
            notes = score.to_beats(analysis, written)
            if call.function.name == "submit_take":
                take = {"notes": notes, "written": written, "level_lu": float(args.get("level_lu", -6)),
                        "to_producer": args.get("to_producer", ""), "checks": checks, "engineer": reviews}
                if engineer is None:
                    return take
                ok, verdicts = engineer(take)
                reviews.append(verdicts)
                if ok or len(reviews) > revisions:
                    return take
                msgs.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(
                    {"status": "sent back by the session engineer before the producer heard it",
                     "verdicts": verdicts}, default=float)})
                continue
            report = score.fit(analysis, notes, drums=drums)
            checks.append(report)
            msgs.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(report)})
    raise RuntimeError("musician never submitted a take")
