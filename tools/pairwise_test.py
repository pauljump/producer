"""Can Spark Open tell a good take from a broken one when it hears both? Forced choice, both orders."""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from producer import hands, selfcheck  # noqa: E402
from producer.ears import parse_json  # noqa: E402

PAIR = """Clip A and clip B are the same passage of a recording, each with a {instrument} part added.
One of the two {instrument} parts has a problem: it is {problem}. Listen to both.
Which clip has the problem? JSON only: {{"answer": "A" or "B", "confidence": 0-1, "why": "short"}}"""
PROBLEMS = {"wrong key (+1)": "out of tune with the guitar (sour, clashing)",
            "late 150ms": "late against the guitar strums",
            "random timing": "sloppy, not locked to the guitar's rhythm"}
song_path, take_path = Path(sys.argv[1]), Path(sys.argv[2])
WHO = sys.argv[3] if len(sys.argv) > 3 else "spark-open"
import base64, tempfile  # noqa: E402
from producer import models  # noqa: E402


def meta_mp3(clip):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.mp3"
        hands.save(p, clip)
        return {"type": "input_audio", "input_audio": {"data": base64.b64encode(p.read_bytes()).decode(), "format": "mp3"}}


def judge(prompt, ca, cb, purpose):
    if WHO == "spark-open":
        return selfcheck.ask([{"type": "text", "text": prompt}, {"type": "text", "text": "Clip A:"}, selfcheck._mp3(ca),
                              {"type": "text", "text": "Clip B:"}, selfcheck._mp3(cb)], purpose=purpose)
    model = {"wonder": models.WONDER, "spark-1.2": models.EARS}[WHO]
    r = models.chat("meta", model, [{"role": "system", "content": selfcheck.SYSTEM}, {"role": "user", "content": [
        {"type": "text", "text": prompt}, {"type": "text", "text": "Clip A:"}, meta_mp3(ca),
        {"type": "text", "text": "Clip B:"}, meta_mp3(cb)]}], purpose=purpose + f" ({WHO})", estimate_usd=0.15, max_tokens=8000)
    return models.text(r)
a = json.loads(song_path.with_suffix(".analysis.json").read_text())
t = json.loads(take_path.read_text())
inst, notes, level = t["musician"]["instrument"], t["take"]["notes"], t["take"]["level_lu"]
random.seed(7)
broken = {"wrong key (+1)": [{**n, "pitch": n["pitch"] + 1} for n in notes],
          "late 150ms": [{**n, "beat": n["beat"] + 0.15 * a["tempo_bpm"] / 60} for n in notes],
          "random timing": [{**n, "beat": n["beat"] + random.uniform(-0.3, 0.3)} for n in notes]}
song = hands.load(song_path)
def mixed(ns):
    return hands.mix(song, [(hands.render({"instrument": inst, "notes": ns}, a["beats"], a["duration"]), level)])
good = mixed(notes)
score = {}
for name, ns in broken.items():
    bad = mixed(ns)
    right = 0
    trials = 0
    for s, e in [(0, 15), (18, 33), (36, 51)]:
        x, y = int(s * hands.SR), int(e * hands.SR)
        for bad_is in "AB":
            clips = (good[x:y], bad[x:y]) if bad_is == "B" else (bad[x:y], good[x:y])
            text = judge(PAIR.format(instrument=inst, problem=PROBLEMS[name]), clips[0], clips[1],
                         purpose=f"self-check pairwise test: {name}")
            try:
                ans = parse_json(text).get("answer")
            except ValueError:
                ans = None
            trials += 1
            right += ans == bad_is
    score[name] = f"{right}/{trials}"
    print(f"{name:16s} correct {right}/{trials}")
take_path.with_suffix(f".pairwise-{WHO}.json").write_text(json.dumps(score, indent=1))
