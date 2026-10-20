"""Fruit Ripeness Detector - Flask backend (YOLOv8n)."""
import os
import subprocess
import tempfile
import time
import uuid
from collections import defaultdict

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory
from ultralytics import YOLO

# ---------- 1. SETTINGS ----------
MODEL_PATH = os.environ.get("MODEL_PATH", "best.pt")   # the trained model in this folder
IMG_SIZE = 416            # must match the size used in training
CONF = 0.35               # minimum confidence for a detection to be shown
MAX_SECONDS = 20          # only process the first 20 s of a video
MAX_WIDTH = 640           # shrink wide frames to keep processing fast
MAX_UPLOAD_MB = 30        # Cloud Run limits uploads to about 32 MiB
ALLOWED_VIDEO = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
WORK_DIR = os.path.join(tempfile.gettempdir(), "fruit_app")   # temporary files
os.makedirs(WORK_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# ---------- 2. LOAD THE MODEL ONCE, WHEN THE APP STARTS ----------
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names          # e.g. {0: 'Ripe', 1: 'Rotten', ...}


def cleanup_old_files(max_age_s=3600):
    """Delete temporary files older than an hour so the server does not fill up."""
    now = time.time()
    for name in os.listdir(WORK_DIR):
        path = os.path.join(WORK_DIR, name)
        if os.path.isfile(path) and now - os.path.getmtime(path) > max_age_s:
            try:
                os.remove(path)
            except OSError:
                pass


def count_classes(result):
    """Count how many boxes of each class are in one frame."""
    counts = defaultdict(int)
    for c in result.boxes.cls.tolist():
        counts[CLASS_NAMES[int(c)]] += 1
    return counts


# ---------- 3. PAGES ----------
@app.route("/")
def index():
    return render_template("index.html", class_names=list(CLASS_NAMES.values()))


@app.route("/health")
def health():
    return jsonify(status="ok", classes=list(CLASS_NAMES.values()))


# ---------- 4. VIDEO PREDICTION ----------
@app.route("/predict_video", methods=["POST"])
def predict_video():
    cleanup_old_files()
    file = request.files.get("video")
    if not file or file.filename == "":
        return jsonify(error="No video uploaded."), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_VIDEO:
        return jsonify(error=f"Unsupported file type {ext}. Use mp4/mov/avi/mkv/webm."), 400

    # save the upload to a temporary file
    uid = uuid.uuid4().hex[:10]
    in_path = os.path.join(WORK_DIR, f"in_{uid}{ext}")
    raw_path = os.path.join(WORK_DIR, f"raw_{uid}.mp4")
    out_name = f"out_{uid}.mp4"
    out_path = os.path.join(WORK_DIR, out_name)
    file.save(in_path)

    # open the video and read its properties
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        return jsonify(error="Could not read the video file."), 400
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, MAX_WIDTH / w) if w else 1.0
    out_w, out_h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2   # even sizes for the encoder

    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    max_frames = int(MAX_SECONDS * fps)
    best_counts = defaultdict(int)     # most fruit of each class seen in one frame
    frames_with_det, n = 0, 0

    # go through the video frame by frame
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if scale < 1.0:
            frame = cv2.resize(frame, (out_w, out_h))
        result = model.predict(frame, conf=CONF, imgsz=IMG_SIZE, verbose=False)[0]   # run the model
        counts = count_classes(result)
        if counts:
            frames_with_det += 1
        for cls, k in counts.items():
            best_counts[cls] = max(best_counts[cls], k)
        writer.write(result.plot())    # frame with the boxes and labels drawn on it
        n += 1

    cap.release()
    writer.release()
    os.remove(in_path)
    if n == 0:
        return jsonify(error="The video contained no readable frames."), 400

    # re-encode to H.264 so every browser can play the result
    try:
        subprocess.run(["ffmpeg", "-y", "-i", raw_path, "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", "-loglevel", "error", out_path],
                       check=True, timeout=600)
        os.remove(raw_path)
    except Exception:
        os.replace(raw_path, out_path)   # fallback if ffmpeg is missing (may not play in browsers)

    return jsonify(video_url=f"/results/{out_name}",
                   classes=[{"name": k, "max_in_frame": v} for k, v in sorted(best_counts.items())],
                   frames_processed=n, frames_with_detections=frames_with_det,
                   truncated=n >= max_frames)


# ---------- 5. PHOTO PREDICTION ----------
@app.route("/predict_image", methods=["POST"])
def predict_image():
    cleanup_old_files()
    file = request.files.get("image")
    if not file or file.filename == "":
        return jsonify(error="No image uploaded."), 400
    uid = uuid.uuid4().hex[:10]
    in_path = os.path.join(WORK_DIR, f"img_{uid}.jpg")
    out_name = f"img_out_{uid}.jpg"
    file.save(in_path)
    img = cv2.imread(in_path)
    os.remove(in_path)
    if img is None:
        return jsonify(error="Could not read the image."), 400
    result = model.predict(img, conf=CONF, imgsz=IMG_SIZE, verbose=False)[0]
    counts = count_classes(result)
    cv2.imwrite(os.path.join(WORK_DIR, out_name), result.plot())
    return jsonify(image_url=f"/results/{out_name}",
                   classes=[{"name": k, "max_in_frame": v} for k, v in sorted(counts.items())],
                   frames_processed=1, frames_with_detections=1 if counts else 0)


# ---------- 6. SERVE RESULT FILES, HANDLE ERRORS, START ----------
@app.route("/results/<path:name>")
def results(name):
    return send_from_directory(WORK_DIR, name)


@app.errorhandler(413)
def too_large(e):
    return jsonify(error=f"File too large. The maximum is {MAX_UPLOAD_MB} MB."), 413


if __name__ == "__main__":
    # the cloud platform tells the app which port to use through the PORT variable
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
