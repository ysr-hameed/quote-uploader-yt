#!/usr/bin/env python3
import os
import sys
import json
import time
import random
import logging
import subprocess
import secrets
import schedule
import threading
import re
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps
from flask import Flask, request, jsonify, redirect, Response, session, send_from_directory, url_for
from dotenv import load_dotenv, find_dotenv
from urllib.parse import urlencode, urljoin

# ---------- CONFIG & STARTUP ----------
APP_DIR = Path(__file__).resolve().parent
ENV_PATH = find_dotenv(usecwd=True) or str(APP_DIR / ".env")
load_dotenv(ENV_PATH)

# Where final, publicly served videos live
VIDEO_FOLDER = APP_DIR / "generated_videos"
VIDEO_FOLDER.mkdir(parents=True, exist_ok=True)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("gemini_youtube_opt_fast")

# Lightweight defaults (override in env)
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

# Gemini / Google LLM
API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_URL = os.getenv(
    "GEMINI_MODEL_URL",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
)
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "45"))
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.35"))
GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "1200"))  # long desc + 100+ hashtags

# Facebook / Instagram (Graph API)
FACEBOOK_PAGE_ID = os.getenv("FACEBOOK_PAGE_ID")
FACEBOOK_ACCESS_TOKEN = os.getenv("FACEBOOK_ACCESS_TOKEN")  # **MUST be a PAGE access token**
INSTAGRAM_USER_ID = os.getenv("INSTAGRAM_USER_ID")  # IG Business/Creator user id (linked to the Page)

# OAuth / YouTube
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


def _timeit(name):
    class _Ctx:
        def __enter__(self):
            self.t0 = time.time()
            log.debug("starting: %s", name)
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
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


# ---------- URL helpers ----------
def _public_base_url():
    """Try to construct a public base URL that respects reverse proxy headers."""
    proto = request.headers.get("X-Forwarded-Proto", request.scheme)
    host = request.headers.get("X-Forwarded-Host", request.host)
    return f"{proto}://{host}/"


def build_public_video_url(filename: str) -> str:
    # Serve from our Flask route; must be publicly reachable and HTTPS for IG
    base = _public_base_url()
    return urljoin(base, f"serve-video/{filename}")


# ---------- OAuth helpers ----------
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


# ---------- Gemini prompt (forced JSON) ----------
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


def _strip_code_fences(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.IGNORECASE)
    return text.strip()


def _extract_first_json_object(text: str):
    if not text:
        return None
    text = _strip_code_fences(text)
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start == -1:
        return None
    brace = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            brace += 1
        elif ch == "}":
            brace -= 1
            if brace == 0:
                snippet = text[start:i+1]
                try:
                    return json.loads(snippet)
                except Exception:
                    continue
    return None


def _enforce_schema(d: dict) -> dict:
    if not isinstance(d, dict):
        d = {}
    return {
        "title": str(d.get("title") or "I Rise"),
        "quote": str(d.get("quote") or "I fell, I learned, and now I rise with fearless focus."),
        "youtube_title": str(d.get("youtube_title") or "I Rise | Motivational Quote for Success"),
        "youtube_description": str(d.get("youtube_description") or "I fell, I learned, and now I rise with fearless focus. Subscribe for daily motivation."),
        "youtube_tags": list(d.get("youtube_tags") or ["motivation", "success", "life lessons", "self improvement"])
    }


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
            parts = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
            )
            text = "".join([p.get("text", "") for p in parts]) if parts else ""
            if not text:
                log.warning("Gemini returned empty text; raw=%s", json.dumps(data)[:2000])
                text = json.dumps(data.get("candidates", [{}])[0].get("content", {}))
            try:
                obj = json.loads(_strip_code_fences(text))
            except Exception:
                obj = _extract_first_json_object(text)
            if not obj:
                log.warning("Failed to parse JSON from Gemini text; beginning extraction fallback")
                obj = _extract_first_json_object(text)
            if not obj:
                raise ValueError("No JSON object could be parsed from Gemini response")
            obj = _enforce_schema(obj)
            return obj
        except Exception as e:
            last_err = e
            log.warning("Gemini attempt %d failed: %s", attempt + 1, e)
            time.sleep(0.6 * (attempt + 1))
    log.warning("Gemini call failed after retries, using fallback. Last error: %s", last_err)
    return _enforce_schema({})


def fetch_quote_from_gemini(seed="motivation"):
    prompt = PROMPT_TEMPLATE + f"\nTopic: {seed}\n"
    try:
        parsed = _call_gemini(prompt)
        return parsed
    except Exception as e:
        log.warning("Gemini call failed: %s", e)
    return _enforce_schema({})


# ---------- Fonts ----------
def _load_font(path, size):
    try:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    except Exception:
        log.debug("failed to load font %s", path, exc_info=True)
    return ImageFont.load_default()


# ---------- Fast frame-based video generator ----------
FADE_DELAY = float(os.getenv("FADE_DELAY", "1.5"))
FADE_DURATION = float(os.getenv("FADE_DURATION", "2"))
HOLD_DURATION = float(os.getenv("HOLD_DURATION", "3"))
FADE_OUT_DURATION = float(os.getenv("FADE_OUT_DURATION", "2"))

FRAMES_DIR_BASE = APP_DIR / "tmp" / "frames"


def generate_static_layers(header_text: str, quote_text: str, width=WIDTH, height=HEIGHT):
    header_font_size = max(28, width // 14)
    quote_font_size = max(26, width // 20)
    font_header = _load_font(HEADER_FONT_PATH, header_font_size)
    font_quote = _load_font(QUOTE_FONT_PATH, quote_font_size)

    header_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(header_img)
    header_h = int(height * 0.12)
    draw.rectangle([0, 0, width, header_h], fill=(255, 255, 255, 255))
    bbox = draw.textbbox((0, 0), header_text, font=font_header)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (header_h - th) // 2), header_text, font=font_header, fill=(0, 0, 0))

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

    line_h = getattr(font_quote, "size", quote_font_size) + 12
    total_h = len(lines) * line_h
    y_start = max(int(height * 0.18), (height // 2) - (total_h // 2))
    for i, line in enumerate(lines):
        x = margin_x
        y = y_start + i * line_h
        parts = line.split(" ")
        cursor = x
        for j, p in enumerate(parts):
            clean = p.strip('.,!?:;').lower()
            text_block = p + (" " if j != len(parts) - 1 else "")
            tw = content_draw.textlength(text_block, font=font_quote)
            if clean in {"success", "focus", "rise", "win", "courage", "learn", "fearless", "fight"}:
                pad = 6
                rect_w = content_draw.textlength(p, font=font_quote) + pad * 2
                rect_h = getattr(font_quote, "size", quote_font_size) + 6
                content_draw.rectangle([cursor - 2, y - 2, cursor + rect_w, y + rect_h], fill=(255, 230, 80))
                content_draw.text((cursor + pad, y), p, font=font_quote, fill=(0, 0, 0))
                cursor += rect_w + content_draw.textlength(" ", font=font_quote)
            else:
                content_draw.text((cursor, y), text_block, font=font_quote, fill=(0, 0, 0, 255))
                cursor += tw

    return header_img.convert("RGBA"), content_img.convert("RGBA")


def generate_frames_and_encode(header_img: Image.Image, content_img: Image.Image, out_video: Path,
                               duration: int = DURATION_TOTAL, fps: int = FPS):
    frames_dir = FRAMES_DIR_BASE / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    if out_video.exists():
        out_video.unlink()  # delete old video so new one replaces it

    fade_delay = FADE_DELAY
    fade_in = FADE_DURATION
    fade_out = FADE_OUT_DURATION
    hold = duration - (fade_delay + fade_in + fade_out)
    if hold < 0:
        scale = duration / (fade_delay + fade_in + fade_out)
        fade_delay *= scale
        fade_in *= scale
        fade_out *= scale
        hold = 0

    total_frames = max(1, int(round(duration * fps)))
    frames = total_frames

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
        a = max(0.0, min(1.0, a))
        alphas.append(a)

    base = Image.new("RGBA", (content_img.width, content_img.height), (0, 0, 0, 255))

    start_time = time.time()
    for idx, alpha in enumerate(alphas):
        if alpha <= 0:
            blended = base
        elif alpha >= 1:
            blended = content_img
        else:
            blended = Image.blend(base, content_img, alpha)
        frame = blended.copy()
        frame.alpha_composite(header_img)
        frame_rgb = frame.convert("RGB")
        frame_path = frames_dir / f"frame_{idx:05d}.jpg"
        frame_rgb.save(frame_path, "JPEG", quality=85, optimize=True)
        if (idx + 1) % 10 == 0 or idx == frames - 1:
            elapsed = time.time() - start_time
            fps_now = (idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (frames - (idx + 1)) / (fps_now if fps_now > 0 else 1)
            print(f"[INFO] Frame {idx+1}/{frames} | {fps_now:.2f} fps | ETA {remaining:.1f}s", end="\r")
    print()

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


def upload_facebook_video(video_path: Path, title: str, description: str):
    """
    Upload a video to Facebook Page.
    If FACEBOOK_ACCESS_TOKEN is a User token, try to fetch the Page token automatically.
    """
    if not (FACEBOOK_PAGE_ID and FACEBOOK_ACCESS_TOKEN):
        return {"error": "Missing FACEBOOK_PAGE_ID or FACEBOOK_ACCESS_TOKEN"}

    # Step 1: Use given token, but verify
    page_token = FACEBOOK_ACCESS_TOKEN

    try:
        url_test = f"https://graph.facebook.com/v19.0/{FACEBOOK_PAGE_ID}?fields=id&access_token={page_token}"
        r_test = requests.get(url_test, timeout=10).json()
        if "error" in r_test and r_test["error"].get("code") == 6000:
            # token is likely a User token; fetch Page token
            log.info("Current token cannot upload. Fetching Page token automatically.")
            resp = requests.get(
                f"https://graph.facebook.com/v19.0/me/accounts",
                params={"access_token": FACEBOOK_ACCESS_TOKEN},
                timeout=10
            )
            resp.raise_for_status()
            pages = resp.json().get("data", [])
            for p in pages:
                if str(p.get("id")) == str(FACEBOOK_PAGE_ID):
                    page_token = p.get("access_token")
                    break
            if not page_token:
                return {"error": f"Could not fetch Page token for Page ID {FACEBOOK_PAGE_ID}"}
    except Exception as e:
        log.warning("Page token check/fetch failed: %s", e)
        return {"error": f"Page token check/fetch failed: {e}"}

    # Step 2: Upload the video
    try:
        with open(video_path, "rb") as f:
            files = {"source": f}
            data = {
                "title": title,
                "description": description,
                "published": "true",
                "access_token": page_token,
            }
            url = f"https://graph-video.facebook.com/v19.0/{FACEBOOK_PAGE_ID}/videos"
            resp = SESSION.post(url, files=files, data=data, timeout=600)
            return resp.json()
    except Exception as e:
        log.exception("FB upload exception")
        return {"error": str(e)}

IG_STATUS_FIELDS = "status_code,status,video_status,id"

def create_instagram_reel_container(video_url: str, caption: str):
    url = f"https://graph.facebook.com/v19.0/{INSTAGRAM_USER_ID}/media"
    data = {
        "caption": caption,
        "media_type": "REELS",
        "video_url": video_url,
        "access_token": FACEBOOK_ACCESS_TOKEN,
    }
    r = SESSION.post(url, data=data, timeout=60)
    return r.json()


def wait_for_ig_container(creation_id: str, timeout_s: int = 120, poll_every: float = 3.0):
    """Poll container until FINISHED or ERROR."""
    url = f"https://graph.facebook.com/v19.0/{creation_id}"
    params = {"fields": IG_STATUS_FIELDS, "access_token": FACEBOOK_ACCESS_TOKEN}
    t0 = time.time()
    last = {}
    while time.time() - t0 < timeout_s:
        r = SESSION.get(url, params=params, timeout=30)
        last = r.json()
        status = last.get("status_code") or last.get("status") or last.get("video_status")
        log.info("IG container %s status: %s", creation_id, status)
        if status in ("FINISHED", "FINISH", "READY"):
            return {"ok": True, "last": last}
        if status in ("ERROR", "FAILED"):
            return {"ok": False, "last": last}
        time.sleep(poll_every)
    return {"ok": False, "last": last, "timeout": True}


def publish_instagram_container(creation_id: str):
    url = f"https://graph.facebook.com/v19.0/{INSTAGRAM_USER_ID}/media_publish"
    data = {"creation_id": creation_id, "access_token": FACEBOOK_ACCESS_TOKEN}
    r = SESSION.post(url, data=data, timeout=60)
    return r.json()


def upload_instagram_reel_via_url(video_url: str, caption: str):
    if not (INSTAGRAM_USER_ID and FACEBOOK_ACCESS_TOKEN):
        return {"error": "Missing INSTAGRAM_USER_ID or FACEBOOK_ACCESS_TOKEN"}
    try:
        container = create_instagram_reel_container(video_url, caption)
        if "id" not in container:
            return {"step": "create_container", "response": container}
        creation_id = container["id"]
        wait = wait_for_ig_container(creation_id)
        if not wait.get("ok"):
            return {"step": "wait_container", "response": wait}
        publish = publish_instagram_container(creation_id)
        return {"step": "publish", "response": publish}
    except Exception as e:
        log.exception("IG upload exception")
        return {"error": str(e)}


# ---------- YouTube helpers (resumable upload) ----------
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
            "snippet": {
                "title": title,
                "description": description,
                "tags": (tags or [])[:15]
            },
            "status": {"privacyStatus": privacyStatus}
        }
        init_headers = headers.copy()
        init_headers.update({"Content-Type": "application/json; charset=UTF-8"})
        r = SESSION.post(YOUTUBE_UPLOAD_URL, headers=init_headers, params=params, json=metadata, timeout=30)
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


# ---------- Flask endpoints ----------
@app.route("/health", methods=["GET"])
def health():
    ok = True
    msg = "Service running"
    if not shutil.which("ffmpeg"):
        ok = False
        msg += " (WARNING: ffmpeg missing)"
    return {"status": "ok" if ok else "warn", "message": msg}, 200


@app.get("/debug/gemini")
def debug_gemini():
    topic = request.args.get("topic", "motivation")
    prompt = PROMPT_TEMPLATE + f"\nTopic: {topic}\n"
    obj = _call_gemini(prompt)
    return jsonify({"ok": True, "parsed": obj})


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
          <a href='/channels'>List My Channels</a> |
          <a href='/debug/gemini?topic=motivation'>Test Gemini JSON</a>
        </p>
        <h2>Generate & Upload (fast frames)</h2>
        <form action='/generate-and-upload' method='get'>
          Topic: <input name='topic' value='motivation' />
          Privacy: <select name='privacy'><option>public</option><option>unlisted</option><option>private</option></select>
          Duration (sec): <input name='duration' value='{DURATION_TOTAL}' style='width:80px' />
          <label><input type='checkbox' name='no_upload' value='1'/> Skip upload</label>
          <button type='submit'>Generate</button>
        </form>
        <p>Fast generator uses pre-rendered frames + ffmpeg for faster, consistent output.</n>
        <hr/>
        <p>Videos are served from: <code>{VIDEO_FOLDER}</code></p>
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
        _ = exchange_code_for_tokens(code)
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
    topic = request.args.get('topic', 'motivation')
    privacy = request.args.get('privacy', 'public')
    skip_upload = request.args.get('no_upload') == '1'
    duration = int(request.args.get('duration', str(DURATION_TOTAL)))
    duration = max(3, min(MAX_DURATION, duration))

    if not API_KEY:
        return jsonify({'ok': False, 'error': 'GEMINI_API_KEY missing'}), 400

    # Fetch quote + YouTube metadata
    gemini_data = fetch_quote_from_gemini(topic)
    quote_title = gemini_data['title']
    quote_text = gemini_data['quote']
    youtube_title = gemini_data['youtube_title']
    youtube_description = gemini_data['youtube_description']
    youtube_tags = gemini_data['youtube_tags']

    tmpdir = APP_DIR / "tmp" / "video"
    tmpdir.mkdir(parents=True, exist_ok=True)

    header_img, content_img = generate_static_layers(quote_title, quote_text, width=WIDTH, height=HEIGHT)
    out_video_tmp = tmpdir / "out_no_audio.mp4"
    out_video_final_tmp = tmpdir / "out_final.mp4"

    try:
        generate_frames_and_encode(header_img, content_img, out_video_tmp, duration=duration, fps=FPS)
    except Exception as e:
        log.exception("video generation failed")
        return jsonify({'ok': False, 'error': str(e)})

    final_tmp = out_video_tmp
    audio = pick_random_audio()
    if audio:
        try:
            mux_audio_to_video(out_video_tmp, out_video_final_tmp, audio)
            final_tmp = out_video_final_tmp
            try:
                out_video_tmp.unlink()
            except Exception:
                pass
        except Exception as e:
            log.warning("audio mux failed: %s", e)
            final_tmp = out_video_tmp

    # Move to public VIDEO_FOLDER with a unique name
    unique_name = f"reel_{int(time.time())}_{secrets.token_hex(3)}.mp4"
    public_path = VIDEO_FOLDER / unique_name
    shutil.move(str(final_tmp), str(public_path))

    result = {
        'ok': True,
        'generated': {
            'title': quote_title,
            'quote': quote_text,
            'youtube_title': youtube_title,
            'youtube_description': youtube_description,
            'youtube_tags': youtube_tags
        },
        'file': str(public_path),
        'public_url': build_public_video_url(unique_name)
    }

    # ---------- UPLOAD SECTION ----------
    if not skip_upload:
        # YouTube Upload
        try:
            upload_resp = upload_video_resumable(public_path, youtube_title, youtube_description, youtube_tags, privacyStatus=privacy)
            result['upload_youtube'] = upload_resp
        except Exception as e:
            log.warning("YouTube upload failed: %s", e)
            result['upload_youtube_error'] = str(e)

        # Facebook Upload (Page video)
        try:
            fb_resp = upload_facebook_video(public_path, youtube_title, youtube_description)
            result['upload_facebook'] = fb_resp
        except Exception as e:
            log.warning("Facebook upload failed: %s", e)
            result['upload_facebook_error'] = str(e)

        # Instagram Upload (Reels via public URL)
        try:
            ig_resp = upload_instagram_reel_via_url(result['public_url'], youtube_title)
            result['upload_instagram'] = ig_resp
        except Exception as e:
            log.warning("Instagram upload failed: %s", e)
            result['upload_instagram_error'] = str(e)

    return jsonify(result)


@app.post('/upload')
def upload_endpoint():
    if 'video' not in request.files:
        return jsonify({'ok': False, 'error': 'no file uploaded'})

    f = request.files['video']
    title = request.form.get('title') or None
    description = request.form.get('description') or None
    tags_input = request.form.get('tags') or None
    privacy = request.form.get('privacy') or 'public'

    tags = [t.strip() for t in tags_input.split(',')] if tags_input else []

    if not title or not description or not tags:
        try:
            gemini_data = fetch_quote_from_gemini("motivation")
            if not title:
                title = gemini_data.get('youtube_title') or f.filename or "Upload"
            if not description:
                description = gemini_data.get('youtube_description') or ""
            if not tags:
                tags = gemini_data.get('youtube_tags') or []
        except Exception as e:
            log.warning("Gemini SEO generation failed: %s", e)
            title = title or f.filename or "Upload"
            description = description or ""
            tags = tags or []

    # Save uploaded file directly to public folder so it's servable
    public_name = f"upload_{int(time.time())}_{secrets.token_hex(3)}_{f.filename}"
    out_path = VIDEO_FOLDER / public_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f.save(out_path)

    try:
        res = upload_video_resumable(out_path, title, description, tags, privacyStatus=privacy)
        return jsonify({'ok': True, 'file': str(out_path), 'public_url': build_public_video_url(public_name), 'upload': res})
    except Exception as e:
        log.warning('upload failed or not authorized: %s', e)
        return jsonify({'ok': False, 'file': str(out_path), 'public_url': build_public_video_url(public_name), 'error': str(e)})


@app.route("/serve-video/<path:filename>")
def serve_video(filename):
    # Allow range requests would be nicer, but simple send works for IG fetches
    return send_from_directory(VIDEO_FOLDER, filename, as_attachment=False)










# ----------------- CONFIG -----------------
GENERATE_URL = "https://quote-uploader-yt.onrender.com/generate-and-upload"

# ----------------- FLASK ROUTES -----------------

# ----------------- SCHEDULER FUNCTION -----------------
def run_job():
    print(f"[{datetime.now()}] Running video upload job...")
    try:
        response = requests.get(GENERATE_URL, timeout=600)  # 10 min timeout
        print(f"[{datetime.now()}] Job response: {response.text}")
    except Exception as e:
        print(f"[{datetime.now()}] Job error: {e}")

def schedule_jobs():
    # Schedule 5 times per day
    schedule.every().day.at("08:00").do(run_job)
    schedule.every().day.at("11:00").do(run_job)
    schedule.every().day.at("14:00").do(run_job)
    schedule.every().day.at("17:00").do(run_job)
    schedule.every().day.at("20:00").do(run_job)

    while True:
        schedule.run_pending()
        time.sleep(30)








# ---------- Run server ----------
if __name__ == "__main__":
  scheduler_thread = threading.Thread(target=schedule_jobs)
  scheduler_thread.daemon = True  # exits when main thread exits
  scheduler_thread.start()
  
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
