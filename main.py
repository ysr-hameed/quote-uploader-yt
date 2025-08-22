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
from flask import Flask, request, jsonify, redirect, Response, session, send_from_directory, url_for
from dotenv import load_dotenv, find_dotenv

# ---------- CONFIG & STARTUP ----------
APP_DIR = Path(__file__).resolve().parent
ENV_PATH = find_dotenv(usecwd=True) or str(APP_DIR / ".env")
load_dotenv(ENV_PATH)

VIDEO_FOLDER = Path("generated_videos")
VIDEO_FOLDER.mkdir(exist_ok=True)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("gemini_youtube_opt_fast")

# Lightweight defaults (override in env)
WIDTH = int(os.getenv("WIDTH", "720"))
HEIGHT = int(os.getenv("HEIGHT", "1280"))
FPS = int(os.getenv("FPS", "25"))
DURATION_TOTAL = int(os.getenv("DURATION", "8"))
MAX_DURATION = int(os.getenv("MAX_DURATION", "60"))
OUTPUT_VIDEO = Path(os.getenv("OUTPUT_VIDEO", str(APP_DIR / "output.mp4")))
MUSIC_DIR = Path(os.getenv("MUSIC_DIR", str(APP_DIR / "music")))

HEADER_FONT_PATH = os.getenv("HEADER_FONT", str(APP_DIR / "Ubuntu-Bold.ttf"))
QUOTE_FONT_PATH = os.getenv("QUOTE_FONT", str(APP_DIR / "Ubuntu-Regular.ttf"))
BG_IMAGE = os.getenv("BG_IMAGE", str(APP_DIR / "bg.jpg"))

# Gemini / Google LLM
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_URL = os.getenv(
    "GEMINI_MODEL_URL",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
)
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "45"))
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.35"))
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "1200"))

# Facebook/Instagram
FACEBOOK_PAGE_ID = os.getenv("FACEBOOK_PAGE_ID")
FACEBOOK_ACCESS_TOKEN = os.getenv("FACEBOOK_ACCESS_TOKEN")
INSTAGRAM_USER_ID = os.getenv("INSTAGRAM_USER_ID")

# Google OAuth / YouTube
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
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(16))

# ---------- Helper utilities ----------
def _make_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "gemini-youtube-opt-fast/1.0"})
    return s

SESSION = _make_session()

class _timeit:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.t0 = time.time()
        log.debug("starting: %s", self.name)
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.t0
        log.info("%s took %.2f sec", self.name, elapsed)

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

# ---------- OAuth helpers ----------
def _build_auth_url(state: str):
    from urllib.parse import urlencode
    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": OAUTH_REDIRECT_URI,
        "response_type": "code",
        "scope": " ".join(SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
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

# ---------- Gemini JSON prompt ----------
PROMPT_TEMPLATE = """
You are a generator that returns ONLY JSON. No code fences, no commentary.

Generate 1 original, highly engaging short motivational quote in first-person style and return strictly in JSON format with no extra text, explanations, or commentary. The quote must:

- Be 1–2 lines long not too much long keep it shorter and very deep.
- Use very simple, clear, and powerful English that immediately connects with anyone reading not any hard words daily life using words.
- Show a vivid contrast (e.g., failure vs. success, struggle vs. breakthrough, fear vs. courage).
- Fully Humanized and never use any hard word easy to understad words daily life using words.
- Evoke strong emotion and a sense of unstoppable motivation.
- End with a positive, uplifting, and inspiring outcome that sparks action.
- Focus on motivation, success, entrepreneurship, or life lessons.
- Avoid copying any existing quotes; make it unique and memorable.
- Use mindset for quotes like top bewt motivational speakers.
- Use first-person words (I, me, my) to make it personal, relatable, and human-like.
- Include a short, punchy, attention-grabbing title in first-person style that reflects the essence of the quote two people understad quote topic and have best hook to user never skip and relatable.

Constraints:
- "title": Short, humanized version of the quote (<100 chars).
- "youtube_title": Highly clickable, SEO-friendly, in simple english <100 chars.
- "youtube_description": Long, natural, includes:
  - The quote
  - Brief story-like summary that connects with viewers
  - Friendly CTA
  - These details woven in naturally:
      Name: Yasir Hameed
      Email: ysr.hameed.yh@gmail.com
      Instagram: @ysr_hameed
      LinkedIn: https://www.linkedin.com/in/yasir-hameed-59b70b36b?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app
      Facebook: https://www.facebook.com/share/16nphS1fR7/
      Website: https://ysr.free.nf
      Channel Name: Motivation Hub
      Channel Description: My channel shares motivational quotes, life lessons, success tips, and short videos for personal growth.
  - End with 100+ relevant hashtags (motivation, success, entrepreneurship, life lessons, personal growth).
- "youtube_tags": 12–20 highly relevant, SEO-friendly tags.

Return strictly in this JSON shape:
{
  "title": "<quote title>",
  "quote": "<the motivational quote>",
  "youtube_title": "<Humanized, SEO optimized YouTube title>",
  "youtube_description": "<Long, engaging, humanized YouTube description including all details and 100+ hashtags>",
  "youtube_tags": ["tag1", "tag2", "..."]
}
"""

def _call_gemini(prompt: str, timeout: int = GEMINI_TIMEOUT, retries: int = 2):
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": GEMINI_TEMPERATURE,
            "maxOutputTokens": GEMINI_MAX_TOKENS,
            "responseMimeType": "application/json"
        }
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": API_KEY}
    last_err = None
    for attempt in range(retries + 1):
        try:
            with _timeit("gemini_call"):
                r = SESSION.post(MODEL_URL, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = "".join([p.get("text","") for p in parts]) if parts else ""
            if not text:
                log.warning("Gemini returned empty text; raw=%s", json.dumps(data)[:500])
                text = json.dumps(data.get("candidates", [{}])[0].get("content", {}))
            try:
                obj = json.loads(text)
            except Exception:
                # fallback to extraction
                m = re.search(r"\{.*\}", text, flags=re.DOTALL)
                obj = json.loads(m.group(0)) if m else {}
            return obj
        except Exception as e:
            last_err = e
            log.warning("Gemini attempt %d failed: %s", attempt+1, e)
            time.sleep(0.5)
    log.warning("Gemini call failed after retries: %s", last_err)
    return {}

def fetch_quote_from_gemini(seed="motivation"):
    prompt = PROMPT_TEMPLATE + f"\nTopic: {seed}\n"
    try:
        return _call_gemini(prompt)
    except Exception as e:
        log.warning("Gemini call failed: %s", e)
        return {}

# ---------- Video generation ----------
FRAMES_DIR_BASE = APP_DIR / "tmp" / "frames"

def generate_frames_and_encode(header_img, content_img, out_video, duration=DURATION_TOTAL, fps=FPS):
    frames_dir = FRAMES_DIR_BASE / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    if out_video.exists():
        out_video.unlink()

    total_frames = max(1, int(round(duration * fps)))
    base = Image.new("RGBA", content_img.size, (0,0,0,255))
    alphas = [min(1.0, max(0.0, i/total_frames)) for i in range(total_frames)]

    for idx, alpha in enumerate(alphas):
        frame = Image.blend(base, content_img, alpha)
        frame.alpha_composite(header_img)
        frame.convert("RGB").save(frames_dir / f"frame_{idx:05d}.jpg", "JPEG", quality=85)
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%05d.jpg"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_video)]
    subprocess.run(cmd, check=True)
    shutil.rmtree(frames_dir)
    return out_video

# ---------- Facebook/Instagram upload ----------
def upload_facebook_video(video_path, title, description):
    url = f"https://graph-video.facebook.com/v19.0/{FACEBOOK_PAGE_ID}/videos"
    files = {'source': open(video_path, 'rb')}
    data = {'title': title, 'description': description, 'access_token': FACEBOOK_ACCESS_TOKEN}
    resp = requests.post(url, files=files, data=data)
    return resp.json()

def upload_instagram_reel(video_url, caption):
    url = f"https://graph.facebook.com/v19.0/{INSTAGRAM_USER_ID}/media"
    data = {'caption': caption, 'media_type': 'REELS', 'video_url': video_url, 'access_token': FACEBOOK_ACCESS_TOKEN}
    resp = requests.post(url, data=data).json()
    if "id" in resp:
        creation_id = resp["id"]
        publish_url = f"https://graph.facebook.com/v19.0/{INSTAGRAM_USER_ID}/media_publish"
        publish_data = {"creation_id": creation_id, "access_token": FACEBOOK_ACCESS_TOKEN}
        publish_resp = requests.post(publish_url, data=publish_data).json()
        return publish_resp
    return resp

# ---------- Flask endpoints ----------
@app.route("/serve-video/<filename>")
def serve_video(filename):
    return send_from_directory(VIDEO_FOLDER, filename, as_attachment=False)

@app.get("/generate-and-upload")
def generate_and_upload():
    topic = request.args.get("topic", "motivation")
    skip_upload = request.args.get("no_upload") == '1'
    duration = max(3, min(MAX_DURATION, int(request.args.get("duration", str(DURATION_TOTAL)))))

    gemini_data = fetch_quote_from_gemini(topic)
    title = gemini_data.get("title", "I Rise")
    quote_text = gemini_data.get("quote", "I fell, I learned, and now I rise.")
    youtube_title = gemini_data.get("youtube_title", title)
    youtube_description = gemini_data.get("youtube_description", quote_text)
    youtube_tags = gemini_data.get("youtube_tags", ["motivation", "success"])

    header_img, content_img = generate_static_layers(title, quote_text)
    out_video = VIDEO_FOLDER / "out_final.mp4"
    generate_frames_and_encode(header_img, content_img, out_video, duration=duration)

    result = {'ok': True, 'file': str(out_video), 'quote': quote_text}

    if not skip_upload:
        try:
            fb_resp = upload_facebook_video(out_video, youtube_title, youtube_description)
            result['upload_facebook'] = fb_resp
        except Exception as e:
            result['upload_facebook_error'] = str(e)
        try:
            public_url = request.host_url.rstrip("/") + url_for("serve_video", filename=out_video.name)
            ig_resp = upload_instagram_reel(public_url, youtube_title)
            result['upload_instagram'] = ig_resp
        except Exception as e:
            result['upload_instagram_error'] = str(e)

    return jsonify(result)

# ---------- Run server ----------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)