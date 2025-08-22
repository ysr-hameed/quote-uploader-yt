#!/usr/bin/env python3
import os
import sys
import json
import time
import random
import logging
import subprocess
import secrets
from pathlib import Path
from datetime import datetime, timedelta
import shutil
import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps
from flask import Flask, request, jsonify, redirect, Response, session, url_for
from dotenv import load_dotenv, find_dotenv

# ---------- CONFIG & STARTUP ----------
APP_DIR = Path(__file__).resolve().parent
ENV_PATH = find_dotenv(usecwd=True) or str(APP_DIR / ".env")
load_dotenv(ENV_PATH)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
log = logging.getLogger("gemini_youtube_opt")

# Lightweight defaults (override in env)
WIDTH = int(os.getenv("WIDTH", "720"))
HEIGHT = int(os.getenv("HEIGHT", "1280"))
FPS = int(os.getenv("FPS", "25"))
DURATION_TOTAL = int(os.getenv("DURATION", "8"))  # not used directly for the staged sequence
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
GEMINI_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", "35"))

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
    s.headers.update({"User-Agent": "gemini-youtube-opt/1.0"})
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

# OAuth helpers (unchanged logic)
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

# ---------- Gemini call & fallback ----------
PROMPT_TEMPLATE = (
    "Generate one original, highly engaging short motivational quote and SEO metadata "
    "strictly in JSON format with no extra text, explanations, or commentary.\n\n"
    "The JSON must follow this exact structure:\n"
    "{\n"
    "  \"quote_title\": \"<short, catchy, first-person title of the quote>\",\n"
    "  \"quote\": \"<1-3 line motivational quote in simple, powerful, first-person words>\",\n"
    "  \"youtube_title\": \"<different SEO YouTube title, 3-7 words, catchy and optimized>\",\n"
    "  \"description\": \"<max 150 characters, motivational and SEO friendly>\",\n"
    "  \"tags\": [\"tag1\", \"tag2\", \"tag3\"]\n"
    "}\n\n"
    "Rules for the quote:\n"
    "- Must be 1–3 lines long.\n"
    "- Very simple daily-life English words only.\n"
    "- Show a vivid contrast (failure vs success, fear vs courage, etc.).\n"
    "- Evoke strong emotion and unstoppable motivation.\n"
    "- End positive, uplifting, and inspiring.\n"
    "- Use first-person style (I, me, my).\n"
    "- Must be unique, not copied.\n\n"
    "Rules for titles:\n"
    "- quote_title: short first-person, directly reflects the quote.\n"
    "- youtube_title: different, catchy, SEO optimized hook to grab attention.\n\n"
    "Rules for description & tags:\n"
    "- description: max 150 chars, motivational and optimized.\n"
    "- tags: max 6, relevant to motivation, life, success, entrepreneurship.\n"
)


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok", "message": "Service running"}, 200
    
def _call_gemini(prompt: str, timeout: int = GEMINI_TIMEOUT):
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required")
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.25, "maxOutputTokens": 250}
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
            # as fallback try 'output' or raw text fields
            text = json.dumps(data)
        return text


def parse_gemini_json_block(text: str):
    import re
    # find the first balanced JSON object using a simple heuristics approach
    start = text.find("{")
    if start == -1:
        return None
    # attempt to extract up to a reasonable length
    candidate = text[start:]
    # try progressively shorter slices until valid JSON
    for end in range(len(candidate), 0, -1):
        try:
            obj = json.loads(candidate[:end])
            return obj
        except Exception:
            continue
    # fallback: try to find simple {...}
    m = re.search(r"\{(.|\s)*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        log.debug("failed to parse gemini JSON block")
        return None

# Basic SEO helpers: craft better youtube_title with keywords
SEO_POWER_WORDS = ["Daily", "Motivation", "Success", "Mindset", "Rise", "Hustle", "Focus", "Win"]

def simple_seo_from_text(seed: str, quote: str):
    seed_kw = seed.title()
    # make a hooky quote_title
    quote_title = f"I Rise — {seed_kw}"
    # craft youtube title with keyword phrases
    youtube_title = f"{seed_kw} Tips: {random.choice(['Daily Motivation','Success Mindset','Win Today'])}"
    base_tags = [seed, "motivation", "inspiration", "shorts", "success", "daily motivation"]
    description = (quote + " — Short motivational video. Subscribe for daily wins.")[:150]
    return {
        "quote_title": quote_title,
        "quote": quote,
        "youtube_title": youtube_title,
        "description": description,
        "tags": base_tags[:6]
    }


def _enhance_seo(seo: dict, seed: str = "motivation"):
    # enforce constraints: youtube_title 3-7 words, tags max 6, description <=150 chars
    yt = seo.get("youtube_title") or seo.get("quote_title") or seed.title()
    words = yt.split()
    if len(words) < 3:
        yt = (yt + " " + "Daily Motivation") if yt else "Daily Motivation"
    # ensure presence of seed keyword
    if seed.title() not in yt:
        yt = f"{seed.title()} - {yt}"
    # truncate to 7 words
    seo["youtube_title"] = " ".join(yt.split()[:7])
    tags = seo.get("tags") or []
    if isinstance(tags, list):
        # ensure seed present
        if seed not in tags:
            tags.insert(0, seed)
        seo["tags"] = tags[:6]
    else:
        seo["tags"] = [seed]
    desc = seo.get("description") or ""
    if len(desc) > 150:
        seo["description"] = desc[:147] + "..."
    # Ensure quote_title has a hook
    qt = seo.get("quote_title") or "I Rise"
    if not qt.lower().startswith("i"):
        seo["quote_title"] = "I " + qt
    return seo


def generate_seo_and_quote(seed="motivation"):
    key = f"seo:{seed}"
    if key in _SIMPLE_CACHE:
        return _SIMPLE_CACHE[key]
    try:
        raw = _call_gemini(PROMPT_TEMPLATE + f"\nTopic: {seed}\n")
        parsed = parse_gemini_json_block(raw)
        if parsed and parsed.get("quote"):
            parsed = _enhance_seo(parsed, seed)
            _SIMPLE_CACHE[key] = parsed
            return parsed
        log.warning("Gemini returned no parsed JSON, falling back")
    except Exception as e:
        log.warning("Gemini failed or unreachable: %s", e)
    fallback_quote = "I fell, I learned, and now I rise with fearless focus."
    fallback = simple_seo_from_text(seed, fallback_quote)
    fallback = _enhance_seo(fallback, seed)
    _SIMPLE_CACHE[key] = fallback
    return fallback

# ---------- Image creation (with highlighted words) ----------
HIGHLIGHT_WORDS = {"success", "focus", "rise", "win", "courage", "learn", "fearless", "fight"}

def _load_font(path, size):
    try:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    except Exception:
        log.debug("failed to load font %s", path, exc_info=True)
    return ImageFont.load_default()


def render_base_images(quote_title: str, quote: str, out_dir: Path):
    """
    Create three images:
      - header-only image on white background (header bar text black)
      - full image with background and quote (no shadow on quote)
      - full image WITHOUT header (used so header can be overlaid and kept static)
    Returns (header_img_path, full_img_path, full_no_header_path)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    header_path = out_dir / "img_header.jpg"
    full_path = out_dir / "img_full.jpg"
    full_no_header = out_dir / "img_full_no_header.jpg"

    # font sizes
    header_size = max(24, WIDTH // 18)
    quote_size = max(20, WIDTH // 22)
    font_h = _load_font(HEADER_FONT_PATH, header_size)
    font_q = _load_font(QUOTE_FONT_PATH, quote_size)

    # --- header-only white background with black text ---
    bg = Image.new("RGBA", (WIDTH, HEIGHT), (255, 255, 255, 255))
    draw = ImageDraw.Draw(bg)
    header_h = int(HEIGHT * 0.10)
    bbox = draw.textbbox((0, 0), quote_title, font=font_h)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((WIDTH - tw) // 2, (header_h - th) // 2), quote_title, font=font_h, fill=(0, 0, 0))
    bg.convert("RGB").save(header_path, "JPEG", quality=85, optimize=True)

    # --- full image (bg image or light default) ---
    if Path(BG_IMAGE).exists():
        try:
            bg2 = Image.open(BG_IMAGE).convert("RGBA")
            bg2 = ImageOps.fit(bg2, (WIDTH, HEIGHT), method=Image.LANCZOS)
        except Exception:
            log.warning("BG present but failed loading; using plain")
            bg2 = Image.new("RGBA", (WIDTH, HEIGHT), (30, 30, 30, 255))
    else:
        bg2 = Image.new("RGBA", (WIDTH, HEIGHT), (30, 30, 30, 255))
    draw2 = ImageDraw.Draw(bg2)

    # header bar (light translucent) - but on full image we draw a translucent header to show contrast
    header_h = int(HEIGHT * 0.10)
    draw2.rectangle([0, 0, WIDTH, header_h], fill=(255, 255, 255, 180))
    bbox = draw2.textbbox((0, 0), quote_title, font=font_h)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw2.text(((WIDTH - tw) // 2, (header_h - th) // 2), quote_title, font=font_h, fill=(10, 10, 10))

    # prepare no-header copy (same bg but without header rectangle/text)
    bg_no_header = bg2.copy()
    draw_no = ImageDraw.Draw(bg_no_header)
    # erase header area by redrawing original background (if BG_IMAGE present, re-fit portion)
    if Path(BG_IMAGE).exists():
        try:
            orig = Image.open(BG_IMAGE).convert("RGBA")
            orig = ImageOps.fit(orig, (WIDTH, HEIGHT), method=Image.LANCZOS)
            bg_no_header.paste(orig.crop((0, 0, WIDTH, header_h)), (0, 0))
        except Exception:
            # fallback to solid fill
            draw_no.rectangle([0, 0, WIDTH, header_h], fill=(245, 245, 245, 255))
    else:
        draw_no.rectangle([0, 0, WIDTH, header_h], fill=(30, 30, 30, 255))

    # word-wrap quote (NO SHADOW) and highlight some words with yellow marker
    margin_x = int(WIDTH * 0.08)
    max_w = WIDTH - margin_x * 2
    words = quote.split()
    lines, cur = [], []
    for w in words:
        test = " ".join(cur + [w]) if cur else w
        if draw2.textlength(test, font=font_q) <= max_w:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    line_h = getattr(font_q, "size", quote_size) + 8
    total_h = len(lines) * line_h
    y = max(header_h + 10, (HEIGHT // 2) - (total_h // 2))

    # helper to draw line with highlights
    def draw_line_with_highlight(draw_obj, text_line, x, y_pos, font):
        parts = text_line.split(" ")
        cursor = x
        for p in parts:
            clean = p.strip('.,!?:;').lower()
            w = p + (" " if p is not parts[-1] else "")
            w_len = draw_obj.textlength(w, font=font)
            if clean in HIGHLIGHT_WORDS:
                # draw yellow rounded rectangle behind the text
                rect_w = draw_obj.textlength(p, font=font) + 8
                rect_h = font.size + 6 if hasattr(font, 'size') else quote_size + 6
                rect_x0 = cursor
                rect_y0 = y_pos
                rect_x1 = rect_x0 + rect_w
                rect_y1 = rect_y0 + rect_h
                draw_obj.rectangle([rect_x0, rect_y0, rect_x1, rect_y1], fill=(255, 230, 80))
                # draw the word in black on top
                draw_obj.text((cursor + 4, y_pos + 2), p, font=font, fill=(0, 0, 0))
                cursor += rect_w
                # add space
                cursor += draw_obj.textlength(" ", font=font)
            else:
                draw_obj.text((cursor, y_pos + 2), p + " ", font=font, fill=(255, 255, 255))
                cursor += draw_obj.textlength(p + " ", font=font)

    # draw on full image (which currently has a translucent header drawn)
    for i, line in enumerate(lines):
        x = margin_x
        yy = y + i * line_h
        draw_line_with_highlight(draw2, line, x, yy, font_q)

    # draw on no-header image (same placement)
    for i, line in enumerate(lines):
        x = margin_x
        yy = y + i * line_h
        draw_line_with_highlight(draw_no, line, x, yy, font_q)

    full_path.parent.mkdir(parents=True, exist_ok=True)
    bg2.convert("RGB").save(full_path, "JPEG", quality=85, optimize=True)
    bg_no_header.convert("RGB").save(full_no_header, "JPEG", quality=85, optimize=True)

    log.info("Saved images: %s, %s, %s", header_path, full_path, full_no_header)
    return header_path, full_path, full_no_header

# ---------- Video maker (segmented approach with fades) ----------

def pick_random_audio():
    if not MUSIC_DIR.exists():
        return None
    files = [p for p in MUSIC_DIR.iterdir() if p.is_file() and p.suffix.lower() in (".mp3", ".m4a", ".wav", ".ogg", ".aac")]
    return random.choice(files) if files else None


def make_staged_video(header_img: Path, full_img: Path, full_no_header: Path, out_video: Path,
                      header_duration=2, full_total=6, fps=FPS):
    """
    Staged video:
      - header_img shown for header_duration (no fade on header)
      - background/quote (from full_no_header) shown for full_total seconds, with fade in/out
      - header image is overlaid on top of the concatenated result so the header never fades
    """
    if full_total <= 4:
        raise ValueError("full_total should be at least 5 to allow fades (we recommend 6)")

    tmp = out_video.parent / "tmp_vid"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    seg0 = tmp / "seg0.mp4"  # header static video
    seg1 = tmp / "seg1.mp4"  # full_no_header with fades

    # segment 0: header static (no fade) produced from header image
    cmd0 = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(header_img),
        "-t", str(header_duration),
        "-vf", f"scale={WIDTH}:{HEIGHT},fps={fps}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        "-movflags", "+faststart",
        str(seg0)
    ]

    # segment 1: full_no_header image with fade in/out
    fade_in_d = min(2, full_total / 3)
    fade_out_d = min(2, full_total / 3)
    fade_out_start = full_total - fade_out_d
    vf = f"scale={WIDTH}:{HEIGHT},fps={fps},fade=t=in:st=0:d={fade_in_d},fade=t=out:st={fade_out_start}:d={fade_out_d}"
    cmd1 = [
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(full_no_header),
        "-t", str(full_total),
        "-vf", vf,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        "-movflags", "+faststart",
        str(seg1)
    ]

    with _timeit("ffmpeg_seg0"):
        proc = subprocess.run(cmd0, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            log.error("ffmpeg seg0 error: %s", proc.stderr.decode(errors="ignore")[:1000])
            raise RuntimeError("ffmpeg seg0 failed")
    with _timeit("ffmpeg_seg1"):
        proc = subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            log.error("ffmpeg seg1 error: %s", proc.stderr.decode(errors="ignore")[:1000])
            raise RuntimeError("ffmpeg seg1 failed")

    list_txt = tmp / "inputs.txt"
    list_txt.write_text(f"file '{seg0.resolve()}'\nfile '{seg1.resolve()}'\n", encoding="utf-8")

    intermediate = tmp / "concat.mp4"
    cmd_concat = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_txt),
        "-c", "copy",
        str(intermediate)
    ]
    with _timeit("ffmpeg_concat"):
        proc = subprocess.run(cmd_concat, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            log.warning("concat copy failed, trying filter_complex re-encode")
            cmd_fc = [
                "ffmpeg", "-y",
                "-i", str(seg0),
                "-i", str(seg1),
                "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p",
                "-c:v", "libx264", "-crf", "23",
                str(intermediate)
            ]
            proc = subprocess.run(cmd_fc, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode != 0:
                log.error("concat fallback failed: %s", proc.stderr.decode(errors="ignore")[:1000])
                raise RuntimeError("concat failed")

    # Now overlay the header image (static) on top of the concatenated video so header never fades
    overlaid = tmp / "with_header.mp4"
    cmd_overlay = [
        "ffmpeg", "-y",
        "-i", str(intermediate),
        "-i", str(header_img),
        "-filter_complex", f"[1:v]scale={WIDTH}:{HEIGHT}[ovr];[0:v][ovr]overlay=0:0:format=auto",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        str(overlaid)
    ]
    with _timeit("ffmpeg_overlay_header"):
        proc = subprocess.run(cmd_overlay, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            log.error("overlay failed: %s", proc.stderr.decode(errors="ignore")[:1000])
            raise RuntimeError("overlay failed")

    # if audio exists, mix audio to the final file (looped/trimmed) using the overlaid video
    audio = pick_random_audio()
    if audio:
        cmd_mux = [
            "ffmpeg", "-y",
            "-i", str(overlaid),
            "-stream_loop", "-1", "-i", str(audio),
            "-shortest",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "128k",
            str(out_video)
        ]
        with _timeit("ffmpeg_mux_audio"):
            proc = subprocess.run(cmd_mux, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode != 0:
                log.error("audio mux failed: %s", proc.stderr.decode(errors="ignore")[:1000])
                raise RuntimeError("audio mux failed")
    else:
        cmd_final = [
            "ffmpeg", "-y",
            "-i", str(overlaid),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
            str(out_video)
        ]
        with _timeit("ffmpeg_final"):
            proc = subprocess.run(cmd_final, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode != 0:
                log.error("final encode failed: %s", proc.stderr.decode(errors="ignore")[:1000])
                raise RuntimeError("final encode failed")

    size_mb = out_video.stat().st_size / (1024 * 1024)
    log.info("Created video %s (%.2f MB)", out_video, size_mb)
    try:
        shutil.rmtree(tmp)
    except Exception:
        pass
    return out_video

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

# ---------- Flask endpoints ----------
@app.get("/")
def index():
    has_tokens = bool(_load_tokens())
    html = f"""
    <html>
      <head><title>Gemini → YouTube Optimized</title></head>
        <body style='font-family: Arial, sans-serif; max-width:900px;margin:30px auto'>
        <h1>Gemini → YouTube Optimized (Enhanced)</h1>
        <p>Authorized: <strong>{'Yes' if has_tokens else 'No'}</strong></p>
        <p>
          <a href='/auth/start'>Authorize YouTube</a> |
          <a href='/auth/revoke'>Revoke Authorization</a> |
          <a href='/channels'>List My Channels</a>
        </p>
        <h2>Generate & Upload (auto-upload)</h2>
        <form action='/generate-and-upload' method='get'>
          Topic: <input name='topic' value='motivation' />
          Privacy: <select name='privacy'><option>public</option><option>unlisted</option><option>private</option></select>
          <button type='submit'>Generate & Upload</button>
        </form>
        <p>Note: this variant keeps the header static (white header with black text) while the background and quote fade in/out. Important words are highlighted like a yellow marker.</p>
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

    seo = generate_seo_and_quote(topic)
    quote_title = seo.get("quote_title") or seo.get("youtube_title") or topic.title()
    quote = seo.get("quote") or "Keep going, you're closer than you think."

    tmpdir = APP_DIR / "tmp" / f"gen_{int(time.time())}"
    header_img, full_img, full_no_header = render_base_images(quote_title, quote, tmpdir)
    out_video = tmpdir / "out.mp4"
    try:
        make_staged_video(header_img, full_img, full_no_header, out_video, header_duration=2, full_total=6, fps=FPS)
    except Exception as e:
        log.exception("video generation failed")
        return jsonify({'ok': False, 'error': str(e)})

    result = {'ok': True, 'generated': seo, 'file': str(out_video)}
    if not skip_upload:
        try:
            upload_resp = upload_video_resumable(out_video, seo.get("youtube_title") or quote_title,
                                                 seo.get("description") or "", seo.get("tags") or [], privacyStatus=privacy)
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

# ---------- Run server ----------
if __name__ == "__main__":
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
