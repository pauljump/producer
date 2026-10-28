"""Call SAM Audio (Meta open weights) on the deployed Modal app, by text prompt.

Stage-1 comparison on Watchtower (docs/SEPARATION-TEST.md): SAM Audio's voice stem agrees with
Demucs (0.85), and it can pull out what Demucs has no stem for (crowd, talking), but its guitar
stem reads the chords worse than Demucs's. So SAM Audio separates the voice and the room, and
Demucs keeps the instruments.
"""
import os
from pathlib import Path

from . import ledger
from .keys import key

PROMPTS = ["singing voice", "crowd noise", "people talking"]
USD_PER_RUN = 0.15  # L40S, about 4 to 5 minutes per 70 s song with three prompts, warm


def separate(song: Path, outdir: Path, prompts: list[str] = PROMPTS) -> dict[str, Path]:
    os.environ.setdefault("MODAL_TOKEN_ID", key("MODAL_TOKEN_ID"))
    os.environ.setdefault("MODAL_TOKEN_SECRET", key("MODAL_TOKEN_SECRET"))
    import modal

    Separator = modal.Cls.from_name("producer-sam-audio", "Separator")
    got = Separator().separate.remote(song.read_bytes(), prompts)
    ledger.record("modal", "facebook/sam-audio-large (L40S)", f"SAM Audio: {', '.join(prompts)}", USD_PER_RUN,
                  note="estimate; reconcile with `modal billing summary`")
    outdir.mkdir(parents=True, exist_ok=True)
    out = {}
    for prompt, wav in got.items():
        p = outdir / f"{prompt.replace(' ', '-')}.wav"
        p.write_bytes(wav)
        out[prompt] = p
    return out
