"""Does the self-check catch planted mistakes? Render a good take and broken copies of it and
let Spark Open (with its measuring tools) judge each, passage by passage.

usage: tools/selfcheck_test.py audio/song.wav audio/takes/<take>.json stems_dir
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from producer import hands, selfcheck  # noqa: E402

song_path, take_path, stems = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
a = json.loads(song_path.with_suffix(".analysis.json").read_text())
t = json.loads(take_path.read_text())
inst, notes, level = t["musician"]["instrument"], t["take"]["notes"], t["take"]["level_lu"]
random.seed(7)
variants = {
    "good": (notes, level),
    "late 150ms": ([{**n, "beat": n["beat"] + 0.15 * a["tempo_bpm"] / 60} for n in notes], level),
    "wrong key (+1)": ([{**n, "pitch": n["pitch"] + 1} for n in notes], level),
    "random timing": ([{**n, "beat": n["beat"] + random.uniform(-0.3, 0.3)} for n in notes], level),
    "too loud (+10 LU)": (notes, level + 10),
}
song = hands.load(song_path)
n = len(song)
def stem(name):
    x = hands.load(stems / f"{name}.wav")
    return x[:n] if len(x) >= n else hands.np.pad(x, ((0, n - len(x)), (0, 0)))
harmony, vocals = stem("guitar") + stem("other"), stem("vocals")
results = {}
for name, (ns, lv) in variants.items():
    part = hands.at_level(song, hands.render({"instrument": inst, "notes": ns}, a["beats"], a["duration"]), lv)
    v = [selfcheck.check(song, part, harmony, vocals, a["beats"], inst, s, e, label=f"planted test: {name}")
         for s, e in [(0, 15), (18, 33), (36, 51)]]
    results[name] = v
    print(f"{name:18s}", " | ".join(f'{"PASS" if x.get("pass") else "FAIL" if x.get("pass") is False else "??"} {x.get("time")}/{x.get("tune")}/{x.get("level")}' for x in v), flush=True)
take_path.with_suffix(".selfcheck-test.json").write_text(json.dumps(results, indent=1, default=float))
