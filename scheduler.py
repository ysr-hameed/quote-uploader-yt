from flask import Flask, jsonify
import requests
import schedule
import threading
import time
from datetime import datetime
import logging
import pytz  # use pytz for Termux/Render compatibility

app = Flask(__name__)

# ----------------- CONFIG -----------------
GENERATE_URL = "https://quote-uploader-yt.onrender.com/generate-and-upload"
IST = pytz.timezone("Asia/Kolkata")

# ----------------- LOGGING CONFIG -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]  # Render/Termux console friendly
)
logger = logging.getLogger(__name__)

# ----------------- HELPER FUNCTIONS -----------------
def now_ist():
    return datetime.now(IST)

def log_next_job_time():
    next_job = schedule.next_run()
    if next_job:
        next_job_ist = pytz.utc.localize(next_job).astimezone(IST)
        logger.info(f"⏭ Next scheduled job at: {next_job_ist.strftime('%Y-%m-%d %H:%M:%S %Z')}")

# ----------------- FLASK ROUTES -----------------
@app.route("/health")
def health():
    logger.info("💓 Health check pinged")
    return jsonify({"status": "alive", "time": now_ist().strftime("%Y-%m-%d %H:%M:%S %Z")})

# ----------------- SCHEDULER FUNCTION -----------------
def run_job():
    start_time = now_ist()
    logger.info("=" * 60)
    logger.info(f"🚀 Starting video upload job at {start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    try:
        response = requests.get(GENERATE_URL)  # No timeout, long job allowed
        duration = (now_ist() - start_time).total_seconds()
        logger.info(f"✅ Job completed in {duration:.2f} sec | "
                    f"Status: {response.status_code} | Response: {response.text[:150]}...")
    except Exception as e:
        duration = (now_ist() - start_time).total_seconds()
        logger.error(f"❌ Job failed after {duration:.2f} sec | Error: {e}", exc_info=True)
    finally:
        logger.info("=" * 60)
        log_next_job_time()

# ----------------- SCHEDULER WITH IST → UTC CONVERSION -----------------
def schedule_jobs():
    # IST times you want the jobs to run
    ist_times = ["08:00", "11:00", "14:00", "17:00", "20:00"]
    
    for ist_time in ist_times:
        hours, minutes = map(int, ist_time.split(":"))
        # Convert IST to UTC
        utc_hour = (hours - 5) % 24
        utc_minute = (minutes - 30) % 60
        if minutes < 30:
            utc_hour = (utc_hour - 1) % 24
        utc_time = f"{utc_hour:02d}:{utc_minute:02d}"
        schedule.every().day.at(utc_time).do(run_job)
        logger.info(f"Scheduled job for IST {ist_time} → UTC {utc_time}")

    logger.info("📅 Scheduler started with 5 daily jobs (IST → UTC conversion)")
    log_next_job_time()

    # Run one job immediately on startup
    run_job()

    heartbeat_counter = 0
    while True:
        schedule.run_pending()
        time.sleep(30)

        # Heartbeat log every 5 minutes
        heartbeat_counter += 30
        if heartbeat_counter >= 300:
            logger.info(f"💤 Still alive at {now_ist().strftime('%Y-%m-%d %H:%M:%S %Z')}")
            log_next_job_time()
            heartbeat_counter = 0

# ----------------- THREADING -----------------
if __name__ == "__main__":
    scheduler_thread = threading.Thread(target=schedule_jobs, daemon=True)
    scheduler_thread.start()

    logger.info("🌐 Flask app started on port 5000")
    app.run(host="0.0.0.0", port=5000, use_reloader=False)