"""SAM Audio (Meta open weights) on a Modal GPU: separate a recording by text prompt.

    modal run modal_app/separate.py --song audio/song.wav --prompts "acoustic guitar,male singing voice,crowd"

Writes <song stem>/sam/<prompt>.wav next to the song. Weights are cached in a Modal volume.
The workspace has a $0 net spend limit, so this only ever runs on credits.
"""
import io
from pathlib import Path

import modal

app = modal.App("producer-sam-audio")
weights = modal.Volume.from_name("producer-sam-audio-weights", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install("torch==2.5.1", "torchaudio==2.5.1", "soundfile", "huggingface_hub")
    .pip_install("git+https://github.com/facebookresearch/sam-audio")
    .env({"HF_HOME": "/weights"})
)
MODEL = "facebook/sam-audio-large"


@app.cls(image=image, gpu="L40S", volumes={"/weights": weights}, timeout=1200, scaledown_window=60,
         secrets=[modal.Secret.from_local_environ(["HF_TOKEN"])])
class Separator:
    @modal.enter()
    def load(self):
        import torch
        from sam_audio import SAMAudio, SAMAudioProcessor

        self.processor = SAMAudioProcessor.from_pretrained(MODEL)
        self.model = SAMAudio.from_pretrained(MODEL).eval().cuda()
        self.torch = torch
        weights.commit()

    @modal.method()
    def separate(self, audio: bytes, prompts: list[str]) -> dict[str, bytes]:
        import tempfile

        import torchaudio

        out = {}
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            f.write(audio)
            f.flush()
            for prompt in prompts:
                batch = self.processor(audios=[f.name], descriptions=[prompt]).to("cuda")
                with self.torch.inference_mode():
                    result = self.model.separate(batch, predict_spans=True, reranking_candidates=4)
                buf = io.BytesIO()
                torchaudio.save(buf, result.target.cpu(), self.processor.audio_sampling_rate, format="wav")
                out[prompt] = buf.getvalue()
        return out


@app.local_entrypoint()
def main(song: str, prompts: str):
    src = Path(song)
    got = Separator().separate.remote(src.read_bytes(), [p.strip() for p in prompts.split(",") if p.strip()])
    outdir = src.parent / src.stem / "sam"
    outdir.mkdir(parents=True, exist_ok=True)
    for prompt, wav in got.items():
        (outdir / f"{prompt.replace(' ', '-')}.wav").write_bytes(wav)
        print("wrote", outdir / f"{prompt.replace(' ', '-')}.wav")
