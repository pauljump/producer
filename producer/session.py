"""A recording session: the song, the band, every take, and what the producer kept.

State lives in sessions/<name>/session.json (gitignored with the audio). The producer's whole
vocabulary is: invite, say what you feel, keep, send back, solo, mute.
"""
import base64
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import analysis as measure
from . import band, ears, hands, score, selfcheck

ROOT = Path(__file__).resolve().parent.parent / "sessions"


class Session:
    def __init__(self, name: str):
        self.dir = ROOT / name
        self.path = self.dir / "session.json"
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {}

    # ---------- setup ----------
    @classmethod
    def create(cls, name: str, song: Path, stems: Path, harmony: list[str] = ("guitar", "other"),
               vocals: str = "vocals") -> "Session":
        s = cls(name)
        s.dir.mkdir(parents=True, exist_ok=True)
        wav = s.dir / "song.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(song), "-ar", str(hands.SR), "-ac", "2", str(wav)], check=True)
        hstems = [stems / f"{h}.wav" for h in harmony]
        a = measure.analyze(wav, hstems)
        (s.dir / "analysis.json").write_text(json.dumps(a))
        heard = ears.listen(wav, a, stems)
        s.state = {"name": name, "song": str(wav), "stems": str(stems), "harmony": list(harmony), "vocals": vocals,
                   "song_map": heard["song_map"], "phrases": heard["phrases"], "musicians": {}, "log": [],
                   "originals": band.cast_originals(heard["song_map"])}
        s.save()
        return s

    def save(self):
        self.path.write_text(json.dumps(self.state, indent=1, default=float))

    def log(self, event: str, **kw):
        self.state["log"].append({"ts": time.strftime("%H:%M:%S"), "event": event, **kw})

    # ---------- shared material ----------
    @property
    def analysis(self) -> dict:
        if not hasattr(self, "_a"):
            self._a = json.loads((self.dir / "analysis.json").read_text())
        return self._a

    @property
    def song(self) -> np.ndarray:
        if not hasattr(self, "_song"):
            self._song = hands.load(Path(self.state["song"]))
        return self._song

    def _stem(self, name: str) -> np.ndarray:
        x, _ = __import__("soundfile").read(str(Path(self.state["stems"]) / f"{name}.wav"), always_2d=True)
        n = len(self.song)
        x = x[:n] if len(x) >= n else np.pad(x, ((0, n - len(x)), (0, 0)))
        return x if x.shape[1] == 2 else np.repeat(x, 2, axis=1)

    def _audio_part(self) -> dict:
        wav16 = subprocess.run(["ffmpeg", "-v", "error", "-i", self.state["song"], "-ar", "16000", "-ac", "1", "-f", "wav", "-"],
                               capture_output=True, check=True).stdout
        return {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav16).decode(), "format": "wav"}}

    def chart(self) -> list[dict]:
        return score.chart(self.analysis, self.state["song_map"])

    # ---------- the engineer ----------
    def engineer(self, instrument: str):
        harmony = sum(self._stem(h) for h in self.state["harmony"])
        vocals = self._stem(self.state["vocals"])
        dur = self.analysis["duration"]
        windows = [(s, min(s + 15.0, dur)) for s in np.arange(0, dur - 5, 15.0)]

        def run(take):
            part = hands.at_level(self.song, self.render_notes(instrument, take["notes"]), take["level_lu"])
            with ThreadPoolExecutor(4) as pool:
                verdicts = list(pool.map(lambda w: selfcheck.check(self.song, part, harmony, vocals, self.analysis["beats"],
                                                                     instrument, w[0], w[1], label="take",
                                                                     chords=self.analysis.get("chords_in_key_per_beat")), windows))
            ok = all(v.get("pass") is not False for v in verdicts)
            return ok, verdicts
        return run

    # ---------- rendering ----------
    def render_notes(self, instrument: str, notes: list[dict]) -> np.ndarray:
        return hands.render({"instrument": instrument, "notes": notes}, self.analysis["beats"], self.analysis["duration"])

    def write_take_audio(self, mid: str, n: int, take: dict) -> dict:
        inst = self.state["musicians"][mid]["character"]["instrument"]
        part = hands.at_level(self.song, self.render_notes(inst, take["notes"]), take["level_lu"])
        base = self.dir / "takes" / f"{mid}-{n}"
        hands.save(base.with_suffix(".solo.mp3"), part)
        hands.save(base.with_suffix(".mix.mp3"), hands.mix(self.song + self.kept_mix(exclude=mid) + part, []))
        return {"solo": str(base.with_suffix(".solo.mp3")), "mix": str(base.with_suffix(".mix.mp3"))}

    def kept_mix(self, exclude: str | None = None, muted: tuple = ()) -> np.ndarray:
        out = np.zeros_like(self.song)
        for mid, m in self.state["musicians"].items():
            if mid == exclude or mid in muted or m.get("kept") is None:
                continue
            take = m["takes"][m["kept"]]
            inst = m["character"]["instrument"]
            out += hands.at_level(self.song, self.render_notes(inst, take["notes"]), take["level_lu"])
        return out

    # ---------- the producer's vocabulary ----------
    def invite(self, ask: str) -> str:
        character = band.cast(ask, self.state["song_map"])
        try:
            character["research"] = band.research(character, self.state["song_map"])
        except Exception as e:  # research is a bonus; the session goes on without it
            character["research"] = {"text": "", "sources": [], "error": str(e)[:200]}
        mid = f'{character["instrument"].replace(" ", "-")}-{character["name"].lower()}'
        self.state["musicians"][mid] = {"character": character, "takes": [], "kept": None, "asked": ask}
        self.log("invite", musician=mid, ask=ask)
        self._play(mid)
        return mid

    def direct(self, mid: str, words: str) -> dict:
        m = self.state["musicians"][mid]
        last = m["takes"][-1]
        instr = band.translate(words, m["character"], last, self.state["song_map"])
        self.log("direct", musician=mid, words=words, instruction=instr)
        direction = f'{instr["instruction"]} Where: {instr["where"]}. Keep: {instr["keep"]}.'
        self._play(mid, direction=direction, previous=last, said=words, translation=instr)
        return instr

    def sketch(self, mid: str, raw: bytes, kind: str, words: str = "") -> dict:
        """The producer hums a line or beatboxes a groove; the musician plays it in character."""
        from . import steer
        m = self.state["musicians"][mid]
        sk = steer.sketch(raw, kind, self.analysis["tempo_bpm"], m["character"]["instrument"])
        if not sk["measured"]:
            raise RuntimeError("couldn't hear an idea in that; try again a little louder")
        said = words or ("(hummed an idea)" if kind == "hum" else "(beatboxed a groove)")
        last = m["takes"][-1]
        instr = band.translate(said, m["character"], last, self.state["song_map"], sketch=sk)
        self.log("sketch", musician=mid, kind=kind, sketch=sk, instruction=instr)
        direction = f'{instr["instruction"]} Where: {instr["where"]}. Keep: {instr["keep"]}. The producer {said}: {json.dumps(sk["measured"])}'
        self._play(mid, direction=direction, previous=last, said=said, translation=instr)
        self.state["musicians"][mid]["takes"][-1]["sketch"] = sk
        self.save()
        return instr

    def keep(self, mid: str):
        m = self.state["musicians"][mid]
        m["kept"] = len(m["takes"]) - 1
        self.log("keep", musician=mid, take=m["kept"])
        self.save()

    def send_back(self, mid: str):
        m = self.state["musicians"][mid]
        self.log("send back", musician=mid, take=len(m["takes"]) - 1)
        m["takes"][-1]["sent_back"] = True
        self.save()

    def _play(self, mid: str, **kw):
        m = self.state["musicians"][mid]
        ch = m["character"]
        said, translation = kw.pop("said", None), kw.pop("translation", None)
        take = band.play(ch, self.analysis, self.state["song_map"], self.chart(), self._audio_part(),
                         engineer=self.engineer(ch["instrument"]), **kw)
        take["producer_said"], take["translation"] = said, translation
        m["takes"].append(take)
        take["audio"] = self.write_take_audio(mid, len(m["takes"]) - 1, take)
        self.log("take", musician=mid, take=len(m["takes"]) - 1, engineer_rounds=len(take["engineer"]))
        self.save()

    def album_art(self) -> Path:
        """The bandleader writes the art brief from what it heard and who played; Muse Image paints it."""
        from . import models
        band_list = [f'{m["character"]["name"]} ({m["character"]["instrument"]})' for m in self.state["musicians"].values()
                     if m.get("kept") is not None]
        people = [f'{o["name"]} ({o["role"]})' for o in self.state.get("originals", [])]
        resp = models.chat("meta", models.WONDER, [{"role": "user", "content":
            "Write one image prompt (under 80 words) for the album cover of this finished session. Make it about the "
            "song's world, not about instruments. No text, no letters, no logos, no real people's likeness. "
            f"Song map: {json.dumps(self.state['song_map'])}. On the record: {people + band_list}. Reply with the prompt only."}],
            purpose="bandleader: album art brief", estimate_usd=0.1, max_tokens=8000, reasoning_effort="low")
        brief = models.text(resp).strip()
        out = self.dir / "cards" / "album.png"
        out.parent.mkdir(exist_ok=True)
        out.write_bytes(models.image(brief, purpose="album art", size="1024x1024"))
        self.state["album"] = {"prompt": brief, "path": str(out)}
        self.save()
        return out

    def final_mix(self, path: Path, muted: tuple = ()) -> Path:
        hands.save(path, hands.mix(self.song + self.kept_mix(muted=muted), []))
        return path
