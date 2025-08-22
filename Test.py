#!/usr/bin/env python3
import os
import sys
import json
import time
import random
import logging
import subprocess
import secrets
import re
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps
from flask import Flask, request, jsonify, redirect, Response, session, url_for
from dotenv import load_dotenv, find_dotenv

---------- CONFIG & STARTUP ----------

APP_DIR = Path(file).resolve().parent
ENV_PATH = find_dotenv(usecwd=True) or str(APP_DIR / ".env")
load_dotenv(ENV_PATH)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("gemini_youtube_opt_fast")

Lightweight defaults (override in env)

WIDTH = int(os.getenv("WIDTH", "720"))
HEIGHT = int(os.getenv("HEIGHT", "1280"))
FPS = int(os.getenv("FPS", "25"))
DURATION_TOTAL = int(os.getenv("DURATION", "8"))  # total seconds for quote screen
MAX_DURATION = int(os.getenv("MAX_DURATION", "60"))
OUTPUT_VIDEO = Path(os.getenv("OUTPUT_VIDEO", str(APP_DIR / "output.mp4")))
MUSIC_DIR = Path(os.getenv("MUSIC_DIR", str(APP_DIR / "music")))

HEADER_FONT_PATH = os.getenv("HEADER_FONT", str(APP_DIR / "Ubuntu-Bold.ttf"))
QUOTE_FONT_PATH = os.getenv("QUOTE_FONT", str(APP_DIR / "Ubuntu-Regular.ttf"))
BG_IMAGE = os.getenv("BG_IMAGE", str(APP_DIR / "bg.jpg"))

Gemini / Google LLM

API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCrM0AzC6PLJCcxr0pIljAUGPV-WAe_PAk")
MODEL_URL = os.getenv(
"GEMINI_MODEL_URL",
"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
)
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "35"))

OAuth / YouTube

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
OAUTH_REDIRECT_URI = os.getenv("OAUTH_REDIRECT_URI", "http://localhost:8000/auth/callback")
TOKEN_FILE = APP_DIR / "tkn.json"
TOKEN_FILE.touch(exist_ok=True)

OAUTH_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
YOUTUBE_UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
YOUTUBE_CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"

SCOPES = [
"https://www.googleapis.com/auth/youtube.upload",
"https://www.googleapis.com/auth/youtube.readonly",
"openid",
"email",
"profile",
]

_SIMPLE_CACHE = {}
app = Flask(name)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(16))

---------- Helper utilities ----------

def _make_session():
s = requests.Session()
s.headers.update({"User-Agent": "gemini-youtube-opt-fast/1.0"})
return s

SESSION = _make_session()

def _timeit(name):
class _Ctx:
def enter(self):
self.t0 = time.time()
log.debug("starting: %s", name)
return self
def exit(self, exc_type, exc_val, exc_tb):
elapsed = time.time() - self.t0
log.info("%s took %.2f sec", name, elapsed)
return _Ctx()

def _load_tokens():
try:
text = TOKEN_FILE.read_text(encoding="utf-8")
return json.loads(text or "{}")
except Exception:
log.exception("failed to read token file, returning empty")
return {}

def _save_tokens(obj):
try:
TOKEN_FILE.write_text(json.dumps(obj, indent=2), encoding="utf-8")
except Exception:
log.exception("failed to save tokens")

OAuth helpers (unchanged logic)

def _build_auth_url(state: str):
params = {
"client_id": GOOGLE_CLIENT_ID,
"redirect_uri": OAUTH_REDIRECT_URI,
"response_type": "code",
"scope": " ".join(SCOPES),
"access_type": "offline",
"prompt": "consent",
"state": state,
}
from urllib.parse import urlencode
return f"{OAUTH_AUTH_URL}?{urlencode(params)}"

def exchange_code_for_tokens(code: str):
with _timeit("exchange_code_for_tokens"):
data = {
"code": code,
"client_id": GOOGLE_CLIENT_ID,
"client_secret": GOOGLE_CLIENT_SECRET,
"redirect_uri": OAUTH_REDIRECT_URI,
"grant_type": "authorization_code",
}
r = SESSION.post(OAUTH_TOKEN_URL, data=data, timeout=15)
r.raise_for_status()
token_resp = r.json()
token_resp["obtained_at"] = int(time.time())
if "expires_in" in token_resp:
token_resp["expires_at"] = token_resp["obtained_at"] + int(token_resp["expires_in"]) - 30
_save_tokens(token_resp)
return token_resp

def refresh_access_token(refresh_token: str):
with _timeit("refresh_access_token"):
data = {
"client_id": GOOGLE_CLIENT_ID,
"client_secret": GOOGLE_CLIENT_SECRET,
"refresh_token": refresh_token,
"grant_type": "refresh_token",
}
r = SESSION.post(OAUTH_TOKEN_URL, data=data, timeout=15)
r.raise_for_status()
token_resp = r.json()
token_resp["obtained_at"] = int(time.time())
if "expires_in" in token_resp:
token_resp["expires_at"] = token_resp["obtained_at"] + int(token_resp["expires_in"]) - 30
existing = _load_tokens()
existing.update(token_resp)
if "refresh_token" not in token_resp and "refresh_token" in existing:
existing["refresh_token"] = existing["refresh_token"]
_save_tokens(existing)
return existing

def ensure_valid_token():
tokens = _load_tokens()
if not tokens:
return None
if tokens.get("expires_at") and int(time.time()) < int(tokens.get("expires_at")):
return tokens.get("access_token")
refresh = tokens.get("refresh_token")
if not refresh:
return None
try:
refreshed = refresh_access_token(refresh)
return refreshed.get("access_token")
except Exception as e:
log.error("Failed to refresh token: %s", e)
return None

---------- Gemini call & parsing ----------

PROMPT_TEMPLATE = (
"Generate one original, highly engaging short motivational quote and return strictly in JSON format "
"with no extra text, explanations, or commentary.\n\n"
"Return JSON:\n"
"{\n"
"  "title": "<catchy first-person title>",\n"
"  "quote": "<1-3 line motivational quote in first-person simple words>"\n"
"}\n"
)

def _call_gemini(prompt: str, timeout: int = GEMINI_TIMEOUT):
if not API_KEY:
raise RuntimeError("GEMINI_API_KEY is required")
payload = {
"contents": [{"parts": [{"text": prompt}]}],
"generationConfig": {"temperature": 0.25, "maxOutputTokens": 200}
}
headers = {"Content-Type": "application/json", "x-goog-api-key": API_KEY}
with _timeit("gemini_call"):
r = SESSION.post(MODEL_URL, headers=headers, json=payload, timeout=timeout)
r.raise_for_status()
data = r.json()
# robust extraction
text = None
try:
text = (
data.get("candidates", [{}])[0]
.get("content", {})
.get("parts", [{}])[0]
.get("text")
)
except Exception:
text = None
if not text:
text = json.dumps(data)
return text

def parse_gemini_json_block_safe(text: str):
"""
Extract the first JSON object from a Gemini response safely,
ignoring unsupported regex constructs like ?R.
"""
# Remove code block markers or other decorations
text = re.sub(r"json|", "", text, flags=re.IGNORECASE).strip()

# Attempt to find the first { ... } block manually  
start = text.find("{")  
if start == -1:  
    log.warning("No JSON object found in Gemini text")  
    return None  

brace_count = 0  
for i, c in enumerate(text[start:], start=start):  
    if c == "{":  
        brace_count += 1  
    elif c == "}":  
        brace_count -= 1  
        if brace_count == 0:  
            candidate = text[start:i+1]  
            try:  
                return json.loads(candidate)  
            except json.JSONDecodeError:  
                # If parse fails, skip unknown characters and retry  
                continue  
log.warning("Failed to parse any JSON block from Gemini response")  
return None

def fetch_quote_from_gemini(seed="motivation"):
try:
raw = _call_gemini(PROMPT_TEMPLATE + f"\nTopic: {seed}\n")
parsed = parse_gemini_json_block_safe(raw)  # <-- use 'raw' here
if parsed and parsed.get("quote"):
return parsed.get("title") or "I Rise", parsed.get("quote")
log.warning("Gemini returned no parsed JSON, using fallback")
except Exception as e:
log.warning("Gemini call failed: %s", e)
# fallback
return "I Rise", "I fell, I learned, and now I rise with fearless focus."

---------- Fonts ----------

def _load_font(path, size):
try:
if Path(path).exists():
return ImageFont.truetype(path, size)
except Exception:
log.debug("failed to load font %s", path, exc_info=True)
return ImageFont.load_default()

---------- Fast frame-based video generator ----------

fade/hold parameters (seconds) - tuned for fast, natural feel

FADE_DELAY = float(os.getenv("FADE_DELAY", "0.5"))         # delay before fade starts
FADE_DURATION = float(os.getenv("FADE_DURATION", "1.2"))   # fade-in duration
HOLD_DURATION = float(os.getenv("HOLD_DURATION", "3"))     # will be computed from total
FADE_OUT_DURATION = float(os.getenv("FADE_OUT_DURATION", "1.2"))

We'll compute HOLD_DURATION later based on DURATION_TOTAL

FRAMES_DIR_BASE = APP_DIR / "tmp" / "frames"

def generate_static_layers(header_text: str, quote_text: str, width=WIDTH, height=HEIGHT):
# choose font sizes relative to width
header_font_size = max(28, width // 14)
quote_font_size = max(26, width // 20)
font_header = _load_font(HEADER_FONT_PATH, header_font_size)
font_quote = _load_font(QUOTE_FONT_PATH, quote_font_size)

# Header layer (white bar with centered text)  
header_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))  
draw = ImageDraw.Draw(header_img)  
header_h = int(height * 0.12)  
draw.rectangle([0, 0, width, header_h], fill=(255, 255, 255, 255))  
bbox = draw.textbbox((0, 0), header_text, font=font_header)  
tw = bbox[2] - bbox[0]  
th = bbox[3] - bbox[1]  
draw.text(((width - tw) // 2, (header_h - th) // 2), header_text, font=font_header, fill=(0, 0, 0))  

# Content layer (background image plus quote text)  
if Path(BG_IMAGE).exists():  
    try:  
        bg = Image.open(BG_IMAGE).convert("RGBA")  
        bg = ImageOps.fit(bg, (width, height), method=Image.LANCZOS)  
    except Exception:  
        log.warning("BG present but failed loading; using solid fill")  
        bg = Image.new("RGBA", (width, height), (30, 30, 30, 255))  
else:  
    bg = Image.new("RGBA", (width, height), (30, 30, 30, 255))  

content_img = bg.copy()  
content_draw = ImageDraw.Draw(content_img)  

# word-wrap quote into lines that fit max width  
margin_x = int(width * 0.08)  
max_w = width - 2 * margin_x  
words = quote_text.split()  
lines = []  
cur = []  
for w in words:  
    test = " ".join(cur + [w]) if cur else w  
    if content_draw.textlength(test, font=font_quote) <= max_w and len(cur) < 10:  
        cur.append(w)  
    else:  
        if cur:  
            lines.append(" ".join(cur))  
        cur = [w]  
if cur:  
    lines.append(" ".join(cur))  

# draw lines centered vertically (below header)  
line_h = getattr(font_quote, "size", quote_font_size) + 12  
total_h = len(lines) * line_h  
y_start = max(int(height * 0.18), (height // 2) - (total_h // 2))  
for i, line in enumerate(lines):  
    x = margin_x  
    y = y_start + i * line_h  
    # simple highlight: highlight words in HIGHLIGHT_WORDS  
    parts = line.split(" ")  
    cursor = x  
    for j, p in enumerate(parts):  
        clean = p.strip('.,!?:;').lower()  
        text_block = p + (" " if j != len(parts) - 1 else "")  
        tw = content_draw.textlength(text_block, font=font_quote)  
        if clean in {"success", "focus", "rise", "win", "courage", "learn", "fearless", "fight"}:  
            # draw rounded-ish rect (simple)  
            pad = 6  
            rect_w = content_draw.textlength(p, font=font_quote) + pad * 2  
            rect_h = getattr(font_quote, "size", quote_font_size) + 6  
            content_draw.rectangle([cursor - 2, y - 2, cursor + rect_w, y + rect_h], fill=(255, 230, 80))  
            content_draw.text((cursor + pad, y), p, font=font_quote, fill=(0, 0, 0))  
            cursor += rect_w + content_draw.textlength(" ", font=font_quote)  
        else:  
            content_draw.text((cursor, y), text_block, font=font_quote, fill=(255, 255, 255))  
            cursor += tw  

return header_img.convert("RGBA"), content_img.convert("RGBA")

def generate_frames_and_encode(header_img: Image.Image, content_img: Image.Image, out_video: Path,
duration: int = DURATION_TOTAL, fps: int = FPS, tmpdir: Path = None):
"""
Fast frame generator:
- Precompute blended frames with alpha computed per frame using simple easing
- Overlay header_img (png alpha) onto each blended frame
- Save frames as PNG or JPG then call ffmpeg to encode
"""
if tmpdir is None:
tmpdir = FRAMES_DIR_BASE / f"gen_{int(time.time())}"
frames_dir = tmpdir
if frames_dir.exists():
shutil.rmtree(frames_dir)
frames_dir.mkdir(parents=True, exist_ok=True)

# compute hold duration to fit total  
fade_delay = FADE_DELAY  
fade_in = FADE_DURATION  
fade_out = FADE_OUT_DURATION  
hold = duration - (fade_delay + fade_in + fade_out)  
if hold < 0:  
    # reduce fade durations proportionally if total too small  
    scale = duration / (fade_delay + fade_in + fade_out)  
    fade_delay *= scale  
    fade_in *= scale  
    fade_out *= scale  
    hold = 0  

total_frames = max(1, int(round(duration * fps)))  
frames = total_frames  

# Precompute per-frame alpha (0..1) for content_img  
alphas = []  
for i in range(frames):  
    t = i / fps  
    if t < fade_delay:  
        a = 0.0  
    elif t < fade_delay + fade_in:  
        a = (t - fade_delay) / fade_in  
    elif t < fade_delay + fade_in + hold:  
        a = 1.0  
    elif t < fade_delay + fade_in + hold + fade_out:  
        a = 1.0 - (t - (fade_delay + fade_in + hold)) / fade_out  
    else:  
        a = 0.0  
    # clamp  
    if a < 0:  
        a = 0.0  
    if a > 1:  
        a = 1.0  
    alphas.append(a)  

# Use a black base to blend from (produces fade-in effect). For performance, prepare base image  
base = Image.new("RGBA", (content_img.width, content_img.height), (0, 0, 0, 255))  

start_time = time.time()  
for idx, alpha in enumerate(alphas):  
    if alpha <= 0:  
        blended = base  
    elif alpha >= 1:  
        blended = content_img  
    else:  
        # blend returns new image (fast C code)  
        blended = Image.blend(base, content_img, alpha)  
    # overlay header on top (alpha composite)  
    frame = blended.copy()  
    frame.alpha_composite(header_img)  
    # convert to RGB for ffmpeg speed (smaller files) — remove alpha  
    frame_rgb = frame.convert("RGB")  
    # use JPEG to reduce disk usage but acceptable quality  
    frame_path = frames_dir / f"frame_{idx:05d}.jpg"  
    frame_rgb.save(frame_path, "JPEG", quality=85, optimize=True)  
    if (idx + 1) % 10 == 0 or idx == frames - 1:  
        elapsed = time.time() - start_time  
        fps_now = (idx + 1) / elapsed if elapsed > 0 else 0  
        remaining = (frames - (idx + 1)) / (fps_now if fps_now > 0 else 1)  
        print(f"[INFO] Frame {idx+1}/{frames} | {fps_now:.2f} fps | ETA {remaining:.1f}s", end="\r")  
print()  

# Encode with ffmpeg (image sequence)  
# Create list file for ffmpeg to be robust across platforms  
list_file = frames_dir / "inputs.txt"  
with list_file.open("w", encoding="utf-8") as f:  
    for i in range(frames):  
        p = frames_dir / f"frame_{i:05d}.jpg"  
        f.write(f"file '{p.resolve()}'\n")  
        # set duration per frame using -r at input is enough for constant framerate; using image2pipe can be fragile.  
# ffmpeg invocation — set framerate and encode  
cmd = [  
    "ffmpeg", "-y",  
    "-framerate", str(fps),  
    "-i", str(frames_dir / "frame_%05d.jpg"),  
    "-c:v", "libx264",  
    "-pix_fmt", "yuv420p",  
    "-crf", "23",  
    str(out_video)  
]  
with _timeit("ffmpeg_encode"):  
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  
    if proc.returncode != 0:  
        log.error("ffmpeg encode failed: %s", proc.stderr.decode(errors="ignore")[:2000])  
        raise RuntimeError("ffmpeg encoding failed")  

# cleanup frames_dir (optional)  
try:  
    shutil.rmtree(frames_dir)  
except Exception:  
    pass  
return out_video

def pick_random_audio():
if not MUSIC_DIR.exists():
return None
files = [p for p in MUSIC_DIR.iterdir() if p.is_file() and p.suffix.lower() in (".mp3", ".m4a", ".wav", ".ogg", ".aac")]
return random.choice(files) if files else None

def mux_audio_to_video(video_path: Path, out_path: Path, audio_path: Path):
# loop audio as necessary and trim to video duration
cmd = [
"ffmpeg", "-y",
"-i", str(video_path),
"-stream_loop", "-1", "-i", str(audio_path),
"-shortest",
"-c:v", "copy",
"-c:a", "aac",
"-b:a", "128k",
str(out_path)
]
with _timeit("ffmpeg_mux_audio"):
proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if proc.returncode != 0:
log.error("audio mux failed: %s", proc.stderr.decode(errors="ignore")[:2000])
raise RuntimeError("audio mux failed")
return out_path

---------- YouTube helpers (resumable upload) ----------

def _auth_headers():
token = ensure_valid_token()
if not token:
raise RuntimeError("Not authorized. Please /auth/start to connect a Google account.")
return {"Authorization": f"Bearer {token}", "Accept": "application/json"}

def list_channels():
headers = _auth_headers()
params = {"part": "snippet,contentDetails,statistics", "mine": "true"}
r = SESSION.get(YOUTUBE_CHANNELS_URL, headers=headers, params=params, timeout=15)
r.raise_for_status()
return r.json()

def upload_video_resumable(video_path: Path, title: str, description: str, tags: list, privacyStatus: str = "public"):
with _timeit("upload_video_resumable"):
headers = _auth_headers()
params = {"uploadType": "resumable", "part": "snippet,status"}
metadata = {
"snippet": {"title": title, "description": description, "tags": tags[:15]},
"status": {"privacyStatus": privacyStatus}
}
init_headers = headers.copy()
init_headers.update({"Content-Type": "application/json; charset=UTF-8"})
r = SESSION.post(YOUTUBE_UPLOAD_URL, headers=init_headers, params=params, json=metadata, timeout=15)
r.raise_for_status()
upload_url = r.headers.get("Location")
if not upload_url:
raise RuntimeError("Failed to obtain resumable upload URL")
total = video_path.stat().st_size
log.info("Uploading %s (%.2f MB) to %s", video_path, total / (1024 * 1024), upload_url)
with video_path.open("rb") as f:
put_headers = {"Content-Length": str(total), "Content-Type": "video/*"}
put_headers.update(headers)
put = SESSION.put(upload_url, headers=put_headers, data=f, timeout=300)
if put.status_code not in (200, 201):
try:
log.error("Upload failed: %s", put.text)
except Exception:
pass
put.raise_for_status()
return put.json()

---------- Flask endpoints ----------

@app.route("/health", methods=["GET"])
def health():
return {"status": "ok", "message": "Service running"}, 200

@app.get("/")
def index():
has_tokens = bool(_load_tokens())
html = f"""
<html>
<head><title>Gemini → YouTube Optimized (Fast)</title></head>
<body style='font-family: Arial, sans-serif; max-width:900px;margin:30px auto'>
<h1>Gemini → YouTube Optimized (Fast)</h1>
<p>Authorized: <strong>{'Yes' if has_tokens else 'No'}</strong></p>
<p>
<a href='/auth/start'>Authorize YouTube</a> |
<a href='/auth/revoke'>Revoke Authorization</a> |
<a href='/channels'>List My Channels</a>
</p>
<h2>Generate & Upload (fast frames)</h2>
<form action='/generate-and-upload' method='get'>
Topic: <input name='topic' value='motivation' />
Privacy: <select name='privacy'><option>public</option><option>unlisted</option><option>private</option></select>
<button type='submit'>Generate & Upload</button>
</form>
<p>Fast generator uses pre-rendered frames + ffmpeg for faster, consistent output.</p>
<hr/>
<p>Token file: <code>{TOKEN_FILE}</code></p>
</body>
</html>
"""
return Response(html, mimetype="text/html")

@app.get('/auth/start')
def auth_start():
if not (GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET and OAUTH_REDIRECT_URI):
return Response("Missing GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET / OAUTH_REDIRECT_URI in env", status=500)
state = secrets.token_urlsafe(16)
session['oauth_state'] = state
return redirect(_build_auth_url(state))

@app.get('/auth/callback')
def auth_callback():
error = request.args.get('error')
if error:
return jsonify({'ok': False, 'error': error})
code = request.args.get('code')
state = request.args.get('state')
if not code or state != session.get('oauth_state'):
return Response("Invalid state or missing code", status=400)
try:
tok = exchange_code_for_tokens(code)
return redirect(url_for('index'))
except Exception as e:
log.exception("Token exchange failed")
return Response(f"Token exchange failed: {e}", status=500)

@app.get('/auth/revoke')
def auth_revoke():
tokens = _load_tokens()
if not tokens:
return redirect(url_for('index'))
access = tokens.get('access_token')
refresh = tokens.get('refresh_token')
for t in (access, refresh):
if not t:
continue
try:
r = SESSION.post('https://oauth2.googleapis.com/revoke', params={'token': t}, timeout=10)
log.info('revoke status: %s', r.status_code)
except Exception as e:
log.warning('failed to revoke token: %s', e)
_save_tokens({})
return redirect(url_for('index'))

@app.get('/channels')
def channels_endpoint():
try:
data = list_channels()
return jsonify({'ok': True, 'channels': data})
except Exception as e:
log.exception('channels failed')
return jsonify({'ok': False, 'error': str(e)})

@app.get('/generate-and-upload')
def generate_and_upload():
# Now auto-uploads by default. If 'no_upload=1' present, skip upload.
topic = request.args.get('topic', 'motivation')
privacy = request.args.get('privacy', 'public')
skip_upload = request.args.get('no_upload') == '1'
duration = int(request.args.get('duration', str(DURATION_TOTAL)))

title, quote = fetch_quote_from_gemini(topic)  
quote_title = title or topic.title()  
quote_text = quote or "Keep going, you're closer than you think."  

tmpdir = APP_DIR / "tmp" / f"gen_{int(time.time())}"  
tmpdir.mkdir(parents=True, exist_ok=True)  
header_img, content_img = generate_static_layers(quote_title, quote_text, width=WIDTH, height=HEIGHT)  
out_video_tmp = tmpdir / "out_no_audio.mp4"  
out_video_final = tmpdir / "out_final.mp4"  
try:  
    generate_frames_and_encode(header_img, content_img, out_video_tmp, duration=duration, fps=FPS, tmpdir=tmpdir / "frames")  
except Exception as e:  
    log.exception("video generation failed")  
    return jsonify({'ok': False, 'error': str(e)})  

# optional audio mux  
audio = pick_random_audio()  
if audio:  
    try:  
        mux_audio_to_video(out_video_tmp, out_video_final, audio)  
        final_video = out_video_final  
        # cleanup interim  
        try:  
            out_video_tmp.unlink()  
        except Exception:  
            pass  
    except Exception as e:  
        log.warning("audio mux failed: %s", e)  
        final_video = out_video_tmp  
else:  
    final_video = out_video_tmp  

result = {'ok': True, 'generated': {'title': quote_title, 'quote': quote_text}, 'file': str(final_video)}  
if not skip_upload:  
    try:  
        # minimal SEO: use quote_title as youtube title, description uses quote text  
        upload_resp = upload_video_resumable(final_video, quote_title, quote_text[:150], [], privacyStatus=privacy)  
        result['upload'] = upload_resp  
    except Exception as e:  
        log.warning("upload failed: %s", e)  
        result['upload_error'] = str(e)  
return jsonify(result)

@app.post('/upload')
def upload_endpoint():
if 'video' not in request.files:
return jsonify({'ok': False, 'error': 'no file uploaded'})
f = request.files['video']
title = request.form.get('title') or f.filename or 'Upload'
description = request.form.get('description') or ''
tags = [t.strip() for t in (request.form.get('tags') or '').split(',') if t.strip()]
privacy = request.form.get('privacy') or 'public'

out_path = APP_DIR / 'tmp' / f.filename  
out_path.parent.mkdir(parents=True, exist_ok=True)  
f.save(out_path)  
try:  
    res = upload_video_resumable(out_path, title, description, tags, privacyStatus=privacy)  
    return jsonify({'ok': True, 'file': str(out_path), 'upload': res})  
except Exception as e:  
    log.warning('upload failed or not authorized: %s', e)  
    return jsonify({'ok': False, 'error': str(e)})

---------- Run server ----------

if name == "main":
if not shutil.which("ffmpeg"):
log.warning("ffmpeg not found in PATH; video creation will fail. Install ffmpeg or add to PATH.")
else:
try:
proc = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
log.info("ffmpeg available: %s", proc.stdout.decode().splitlines()[0] if proc.returncode == 0 else "unknown")
except Exception:
log.debug("ffmpeg check failed", exc_info=True)

port = int(os.getenv("PORT", "8000"))  
log.info("Starting server on 0.0.0.0:%d (WIDTH=%s HEIGHT=%s)", port, WIDTH, HEIGHT)  
app.run(host="0.0.0.0", port=port, debug=False)

In this I want set quite color dark gray and highlight some words with yellow mark like and best title hooked and catching title with some hashtag  in title and best long description with fully filled of viral tags and best description details about me and some links etc many much and not everytime same but some details same and like some my personal details add like my email , instagram, linkedin,Facebook,my website link about my channel etc and many much and intresting youtube title hooked and I want it like video which can easily gets million views so best seo and give me full code

