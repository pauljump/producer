"""The live session: the song loops, the band keeps playing, the producer just reacts.

Taking turns ("say something, wait a minute, hear a new take") breaks the producer's train of
thought. So each player, when they join, writes a palette of different versions of their part
up front (Muse Wonder, in parallel, each checked by the engineer). While the song loops, the
producer talks or taps yes / no. Muse Spark Open on Baseten hears what they said and switches
each player to the version that fits, in about two seconds, at the next bar. What the producer
says yes to seeds new versions in the background (Wonder again), which join the palette while
the music plays. Every note is still written by a model; the router only chooses among them.
"""
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from . import band, models
from .ears import parse_json

IDEAS = """You are {name}, the session {instrument} player. Before the producer starts reacting,
you will prepare {n} genuinely different ways to play your part on this song, so they can hear
options instantly. Make them distinct in feel and density (e.g. sparse and patient, walking,
driving eighths, syncopated, melodic), all true to your character and to the song.

SONG MAP: {song_map}

JSON only: {{"ideas": [{{"label": "2-4 plain words the producer would use, e.g. 'walking and jazzy'",
 "direction": "a specific musical instruction to yourself"}}]}}"""

ROUTER = """You are the bandleader in a live session. The song is looping and the band is playing.
The producer knows nothing about music, only what they like, and reacts out loud. Decide in
an instant what changes. Players can only switch among versions they have already written; if
nothing fits, ask a player to develop a new version (it will arrive in about 30 seconds while
the music keeps playing). Keep what the producer liked. Be decisive and brief.

PLAYERS (current = what is playing now; reactions = what the producer said about each version):
{players}

The producer just said: "{words}"
(They are hearing the {section} right now.)

JSON only:
{{"switch": [{{"musician": "id", "to": version number}}],
 "develop": [{{"musician": "id", "from": version number they liked, "instruction": "what to change"}}],
 "mute": ["id"], "unmute": ["id"],
 "say": "a few words back to the producer, like a bandleader would say in the room"}}"""


def ideas(musician: dict, song_map: dict, n: int = 5) -> list[dict]:
    resp = models.chat("meta", models.WONDER, [{"role": "user", "content": IDEAS.format(
        n=n, name=musician["name"], instrument=musician["instrument"], song_map=json.dumps(song_map))}],
        purpose=f"musician palette ideas: {musician['name']}", estimate_usd=0.1, max_tokens=12000, reasoning_effort="low")
    return parse_json(models.text(resp)).get("ideas", [])[:n]


def build_palette(session, mid: str, n: int = 5) -> None:
    """Write n versions in parallel. Each lands in the palette as soon as it is checked."""
    m = session.state["musicians"][mid]
    todo = ideas(m["character"], session.state["song_map"], n)
    m["pending"] = [i["label"] for i in todo]
    session.save()

    def one(idea):
        session._play(mid, version=f'{idea["label"]}: {idea["direction"]}', label=idea["label"], revisions=1, effort="medium")
        with _lock:
            m["pending"] = [p for p in m.get("pending", []) if p != idea["label"]]
            if m.get("live") is None:
                m["live"] = len(m["takes"]) - 1
            session.save()
    with ThreadPoolExecutor(n) as pool:
        list(pool.map(one, todo))


_lock = threading.Lock()


def develop(session, mid: str, base: int, instruction: str) -> None:
    m = session.state["musicians"][mid]
    label = instruction[:40]
    with _lock:
        m.setdefault("pending", []).append(label)
    session._play(mid, direction=instruction, previous=m["takes"][base], label=label, revisions=1, effort="medium")
    with _lock:
        m["pending"] = [p for p in m.get("pending", []) if p != label]
        session.save()


def _players(session) -> str:
    out = []
    for mid, m in session.state["musicians"].items():
        versions = [{"v": i, "label": t.get("label") or f"take {i + 1}", "what": (t.get("to_producer") or "")[:160],
                     "reactions": t.get("reactions", [])} for i, t in enumerate(m["takes"])]
        out.append({"id": mid, "name": m["character"]["name"], "instrument": m["character"]["instrument"],
                    "current": m.get("live"), "muted": m.get("muted", False), "versions": versions,
                    "being_written": m.get("pending", [])})
    return json.dumps(out)


def react(session, words: str, section: str = "song") -> dict:
    """The producer said something mid-song. Route it in about two seconds (Spark Open)."""
    for mid, m in session.state["musicians"].items():  # their words attach to what is playing now
        if m.get("live") is not None and not m.get("muted"):
            m["takes"][m["live"]].setdefault("reactions", []).append(words)
    resp = models.chat("baseten", models.SELF_CHECK, [{"role": "user", "content": ROUTER.format(
        players=_players(session), words=words, section=section)}], purpose="live router", estimate_usd=0.01, max_tokens=2500)
    try:
        d = parse_json(models.text(resp))
    except ValueError:
        d = {"say": "Got it."}
    ms = session.state["musicians"]
    for s in d.get("switch", []):
        if s.get("musician") in ms and 0 <= int(s.get("to", -1)) < len(ms[s["musician"]]["takes"]):
            ms[s["musician"]]["live"] = int(s["to"])
    for mid in d.get("mute", []):
        if mid in ms:
            ms[mid]["muted"] = True
    for mid in d.get("unmute", []):
        if mid in ms:
            ms[mid]["muted"] = False
    session.log("live", words=words, section=section, decision=d)
    session.save()
    for dv in d.get("develop", []):
        mid = dv.get("musician")
        if mid in ms and ms[mid]["takes"]:
            base = int(dv.get("from", ms[mid].get("live") or 0))
            base = base if 0 <= base < len(ms[mid]["takes"]) else 0
            threading.Thread(target=develop, args=(session, mid, base, dv.get("instruction", words)), daemon=True).start()
    return d
