"""The producer's web app: upload, invite, say what you feel, keep or send back, solo and mute.

    .venv/bin/uvicorn producer.server:app --port 7731

Every slow step (listening, writing a take, the engineer's check) runs in a background thread;
the page polls the session and shows what the band is doing.
"""
import json
import re
import subprocess
import threading
import traceback
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import models
from .session import ROOT, Session

WEB = Path(__file__).resolve().parent.parent / "web"
app = FastAPI(title="Producer")
PRIVATE_HOSTS = ("127.0.0.1", "localhost", "pauls-mac-mini")


def _access_code() -> str | None:
    from .keys import key
    try:
        return key("PRODUCER_ACCESS_CODE")
    except RuntimeError:
        return None


@app.middleware("http")
async def gate(request, call_next):
    """On the tailnet or localhost, open. Anywhere else (the public hostname), every paid action
    is behind an access code, given once and kept in a cookie."""
    from fastapi.responses import HTMLResponse, RedirectResponse
    host = (request.headers.get("host") or "").split(":")[0]
    code = _access_code()
    if host.startswith(PRIVATE_HOSTS) or request.url.path in ("/api/health",):
        return await call_next(request)
    if code is None:
        return JSONResponse({"error": "closed"}, status_code=503)
    if request.query_params.get("code") == code:
        resp = RedirectResponse(str(request.url.remove_query_params("code")))
        resp.set_cookie("producer_code", code, max_age=60 * 60 * 24 * 14, httponly=True, secure=True, samesite="lax")
        return resp
    if request.cookies.get("producer_code") != code:
        return HTMLResponse("<!doctype html><meta name=viewport content='width=device-width'><body style='background:#14110f;"
                            "color:#f3ebe1;font:16px system-ui;display:grid;place-items:center;height:90vh'><form>"
                            "<p>You are the producer. Enter your access code.</p><input name=code autofocus "
                            "style='font-size:18px;padding:10px;border-radius:10px'> <button>Enter</button></form>", 401)
    return await call_next(request)
_locks: dict[str, threading.Lock] = {}


def _status(name: str, text: str | None):
    p = ROOT / name / "status.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"busy": text is not None, "doing": text}))


def _job(name: str, label: str, fn):
    lock = _locks.setdefault(name, threading.Lock())
    if not lock.acquire(blocking=False):
        raise HTTPException(409, "the band is still working")

    def run():
        try:
            _status(name, label)
            fn()
        except Exception as e:  # surfaced to the page
            traceback.print_exc()
            (ROOT / name / "error.json").write_text(json.dumps({"error": str(e)[:500]}))
        finally:
            _status(name, None)
            lock.release()
    (ROOT / name / "error.json").unlink(missing_ok=True)
    threading.Thread(target=run, daemon=True).start()


def _card(s: Session, key: str, prompt: str):
    out = s.dir / "cards" / f"{key}.png"
    if out.exists():
        return
    out.parent.mkdir(exist_ok=True)
    out.write_bytes(models.image(prompt, purpose=f"musician card: {key}"))


def _musician_card(s: Session, mid: str):
    c = s.state["musicians"][mid]["character"]
    _card(s, mid, f"Painted portrait of {c['name']}, a session {c['instrument']} player, holding their instrument, "
                  f"{c['feel']} Warm dim studio light, oil painting texture, head and shoulders, no text, no logos.")


def _original_card(s: Session, i: int, o: dict):
    _card(s, f"original-{i}", f"Painted portrait of {o['name']}, {o['role']}, {o['feel']} Playing live in a small venue, "
                              "warm stage light, oil painting texture, head and shoulders, no text, no logos.")


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "song"


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/api/sessions")
def sessions():
    return [p.parent.name for p in sorted(ROOT.glob("*/session.json"))]


@app.get("/api/session/{name}")
def get(name: str):
    s = Session(name)
    status = json.loads((s.dir / "status.json").read_text()) if (s.dir / "status.json").exists() else {"busy": False}
    err = json.loads((s.dir / "error.json").read_text()) if (s.dir / "error.json").exists() else None
    if not s.state:
        return {"name": name, "status": status, "error": err, "ready": False}
    st = s.state
    musicians = []
    for mid, m in st["musicians"].items():
        takes = [{"n": i, "to_producer": t.get("to_producer"), "producer_said": t.get("producer_said"),
                  "reply": (t.get("translation") or {}).get("reply_to_producer"),
                  "engineer_rounds": len(t.get("engineer", [])), "sent_back": t.get("sent_back", False),
                  "solo": f"/files/{name}/takes/{mid}-{i}.solo.mp3"} for i, t in enumerate(m["takes"])]
        musicians.append({"id": mid, "character": m["character"], "kept": m["kept"], "takes": takes,
                          "card": f"/files/{name}/cards/{mid}.png" if (s.dir / "cards" / f"{mid}.png").exists() else None})
    originals = [{**o, "card": f"/files/{name}/cards/original-{i}.png" if (s.dir / "cards" / f"original-{i}.png").exists() else None}
                 for i, o in enumerate(st.get("originals", []))]
    return {"name": name, "ready": True, "status": status, "error": err, "song": f"/files/{name}/song.mp3",
            "song_map": {k: st["song_map"].get(k) for k in ("title_guess", "tempo_bpm", "key", "feel", "sections", "the_song_is_about")},
            "originals": originals, "musicians": musicians, "log": st["log"][-30:],
            "album": f"/files/{name}/cards/album.png" if (s.dir / "cards" / "album.png").exists() else None,
            "final": f"/files/{name}/final.mp3" if (s.dir / "final.mp3").exists() else None,
            "bandleader": json.loads((s.dir / "bandleader.json").read_text()) if (s.dir / "bandleader.json").exists() else None}


@app.get("/files/{name}/{path:path}")
def files(name: str, path: str):
    p = (ROOT / name / path).resolve()
    if ROOT.resolve() not in p.parents or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.post("/api/session")
async def upload(file: UploadFile = File(...), name: str = Form("")):
    slug = _slug(name or Path(file.filename or "song").stem)
    d = ROOT / slug
    d.mkdir(parents=True, exist_ok=True)
    raw = d / ("upload" + Path(file.filename or "x.wav").suffix)
    raw.write_bytes(await file.read())

    def build():
        wav = d / "upload.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-t", "120", "-ar", "44100", "-ac", "2", str(wav)], check=True)
        _status(slug, "Pulling your recording apart into its pieces")
        stems_root = d / "stems"
        subprocess.run([str(Path(__file__).resolve().parent.parent / ".venv/bin/python"), "-m", "demucs", "-n", "htdemucs_6s",
                        "-d", "cpu", "-o", str(stems_root), str(wav)], check=True, capture_output=True)
        _status(slug, "Listening to your song, phrase by phrase")
        s = Session.create(slug, wav, stems_root / "htdemucs_6s" / "upload")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", s.state["song"], "-b:a", "192k", str(s.dir / "song.mp3")], check=True)
        _status(slug, "Drawing the people on your recording")
        for i, o in enumerate(s.state.get("originals", [])[:4]):
            _original_card(s, i, o)
    _status(slug, "Getting set up")
    _job(slug, "Getting set up", build)
    return {"name": slug}


@app.post("/api/session/{name}/invite")
def invite(name: str, ask: str = Form(...)):
    def run():
        s = Session(name)
        _status(name, f'Finding someone for "{ask}"')
        mid = s.invite(ask)
        _status(name, "Painting their portrait")
        _musician_card(s, mid)
    _job(name, "They're listening to the whole song", run)
    return {"ok": True}


@app.post("/api/session/{name}/{mid}/say")
async def say(name: str, mid: str, words: str = Form(""), voice: UploadFile | None = File(None)):
    if voice is not None:
        raw = await voice.read()
        from .steer import _convert
        wav = _convert(raw, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le")
        words = models.transcribe(wav, len(wav) / 32000)
    if not words.strip():
        raise HTTPException(400, "say something")

    def run():
        Session(name).direct(mid, words)
    _job(name, "They're playing it again", run)
    return {"heard": words}


@app.post("/api/session/{name}/{mid}/sketch")
async def sketch(name: str, mid: str, kind: str = Form("hum"), words: str = Form(""), clip: UploadFile = File(...)):
    if kind not in ("hum", "beatbox"):
        raise HTTPException(400, "hum or beatbox")
    raw = await clip.read()

    def run():
        Session(name).sketch(mid, raw, kind, words)
    _job(name, "They're learning your idea" if kind == "hum" else "They're learning your groove", run)
    return {"ok": True}


@app.post("/api/session/{name}/band")
async def talk_to_band(name: str, words: str = Form(""), voice: UploadFile | None = File(None)):
    if voice is not None:
        from .steer import _convert
        wav = _convert(await voice.read(), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le")
        words = models.transcribe(wav, len(wav) / 32000)
    if not words.strip():
        raise HTTPException(400, "say something")

    def run():
        r = Session(name).ask_band(words)
        (ROOT / name / "bandleader.json").write_text(json.dumps(r, default=str))
    _job(name, "The bandleader is on it", run)
    return {"heard": words}


@app.post("/api/session/{name}/finish")
def finish(name: str):
    def run():
        s = Session(name)
        _status(name, "Mixing the finished song")
        s.final_mix(s.dir / "final.mp3")
        _status(name, "Painting the album cover")
        s.album_art()
    _job(name, "Finishing the song", run)
    return {"ok": True}


@app.post("/api/session/{name}/{mid}/keep")
def keep(name: str, mid: str):
    Session(name).keep(mid)
    return {"ok": True}


@app.post("/api/session/{name}/{mid}/back")
def back(name: str, mid: str):
    Session(name).send_back(mid)
    return {"ok": True}


@app.exception_handler(RuntimeError)
def runtime_error(_, exc: RuntimeError):
    return JSONResponse({"error": str(exc)}, status_code=400)


@app.get("/api/health")
def health():
    return {"ok": True}
