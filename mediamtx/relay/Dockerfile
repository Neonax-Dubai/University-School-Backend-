# NVR -> MediaMTX video-only relay. PyAV bundles its own FFmpeg libraries.
FROM python:3.12-slim
RUN pip install --no-cache-dir "av>=14,<16"
COPY relay.py /relay.py
USER 1000:1000
ENTRYPOINT ["python", "-u", "/relay.py"]
