FROM python:3.11-slim

# system tools: ffmpeg re-encodes the result video, libglib2.0-0 is needed by OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# run the app as an ordinary user, not as administrator
RUN useradd --create-home appuser
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics

# install the packages first, so Docker can reuse this slow step when only the code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app.py, best.pt and templates/ into the image
COPY --chown=appuser:appuser . .
USER appuser

# Cloud Run passes the port to listen on in the PORT variable (8080 is the fallback)
EXPOSE 8080
CMD exec gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 2 --timeout 900 app:app
