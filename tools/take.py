"""Invite a musician on a recording and render the take: solo, and mixed with the song.

usage: tools/take.py audio/song.wav "Invite a bassist" [stems_dir]
"""
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from producer import band, hands, score  # noqa: E402

song = Path(sys.argv[1])
ask = sys.argv[2]
analysis = json.loads(song.with_suffix(".analysis.json").read_text())
song_map = json.loads(song.with_suffix(".ears.json").read_text())["song_map"]
chart = score.chart(analysis, song_map)

wav16 = subprocess.run(["ffmpeg", "-v", "error", "-i", str(song), "-ar", "16000", "-ac", "1", "-f", "wav", "-"],
                       capture_output=True, check=True).stdout
audio_part = {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav16).decode(), "format": "wav"}}

musician = band.cast(ask, song_map)
print("CAST", json.dumps(musician, indent=1))
take = band.play(musician, analysis, song_map, chart, audio_part)
print("TO PRODUCER:", take["to_producer"])
print("CHECKS:", json.dumps(take["checks"], indent=1))
print("NOTES:", len(take["notes"]), "level", take["level_lu"])

mix_in = hands.load(song)
part = hands.render({"instrument": musician["instrument"], "notes": take["notes"]}, analysis["beats"], analysis["duration"])
stamp = time.strftime("%H%M%S")
out = song.parent / "takes" / f"{song.stem}-{musician['instrument'].replace(' ', '-')}-{stamp}"
hands.save(out.with_suffix(".solo.mp3"), part)
hands.save(out.with_suffix(".mix.mp3"), hands.mix(mix_in, [(part, take["level_lu"])]))
out.with_suffix(".json").write_text(json.dumps({"musician": musician, "take": take}, indent=1))
print("WROTE", out)
