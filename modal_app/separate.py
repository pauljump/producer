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
    .pip_install("torch==2.8.0", "torchaudio==2.8.0", "torchvision==0.23.0", "torchcodec==0.7.0",
                 "transformers==4.57.1", "soundfile", "huggingface_hub")
    .pip_install("git+https://github.com/facebookresearch/sam-audio")
    # sam-audio's dependencies pull an old protobuf that breaks the Modal runtime; keep versions matched.
    .pip_install("protobuf>=5.29,<7", "torch==2.8.0", "transformers==4.57.1")
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
    def separate(self, audio: bytes, prompts: list[str], chunk_s: float = 15.0, overlap_s: float = 1.0,
                 rerank: int = 2) -> dict[str, bytes]:
        """Separate in short overlapping chunks (a whole song at once runs a 48 GB GPU out of
        memory) and join them with a short crossfade."""
        import tempfile

        import torchaudio

        torch = self.torch
        wav, sr = torchaudio.load(io.BytesIO(audio))
        sr_out = self.processor.audio_sampling_rate
        hop, ov = int(chunk_s * sr), int(overlap_s * sr)
        out = {}
        for prompt in prompts:
            pieces = []
            for start in range(0, wav.shape[1], hop):
                seg = wav[:, max(0, start - ov): start + hop]
                with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                    torchaudio.save(f.name, seg, sr)
                    batch = self.processor(audios=[f.name], descriptions=[prompt]).to("cuda")
                    with torch.inference_mode():
                        result = self.model.separate(batch, predict_spans=False, reranking_candidates=rerank)
                tgt = result.target[0] if isinstance(result.target, (list, tuple)) else result.target
                pieces.append((start > 0, tgt.cpu().float().reshape(1, -1)))
                torch.cuda.empty_cache()
            ovo = int(overlap_s * sr_out)
            joined = pieces[0][1]
            for has_lead, p in pieces[1:]:
                if has_lead and joined.shape[1] >= ovo and p.shape[1] > ovo:
                    fade = torch.linspace(0, 1, ovo)
                    joined[:, -ovo:] = joined[:, -ovo:] * (1 - fade) + p[:, :ovo] * fade
                    p = p[:, ovo:]
                joined = torch.cat([joined, p], dim=1)
            buf = io.BytesIO()
            torchaudio.save(buf, joined, sr_out, format="wav")
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
