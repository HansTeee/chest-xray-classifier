# Chest X-ray pneumonia classifier: FastAPI (internal :8000) + Streamlit (:7860)
# in one CPU-only image. Runs as non-root UID 1000; only port 7860 needs to be
# public, so it suits single-port PaaS hosts.
#
# Build from the repo root:
#   docker build -f docker/Dockerfile -t chest-xray-classifier .

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# CPU-only torch first, from the PyTorch CPU index (avoids ~2 GB of CUDA libs)
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchvision==0.20.1

COPY requirements-serve.txt /tmp/requirements-serve.txt
RUN pip install -r /tmp/requirements-serve.txt && rm /tmp/requirements-serve.txt

# Run as non-root UID 1000 and keep every cache/config dir it writes to
# under its home, so the container also works on hosts that force a UID.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    TORCH_HOME=/home/user/.cache/torch \
    XDG_CACHE_HOME=/home/user/.cache \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    STREAMLIT_HOME=/home/user/.streamlit \
    API_URL=http://localhost:8000 \
    MODEL_PATH=/home/user/app/models/model.pt \
    MODEL_CARD_PATH=/home/user/app/models/model_card.json \
    TORCH_THREADS=2 \
    MODEL_URL=https://github.com/HansTeee/chest-xray-classifier/releases/download/model-v1/model.pt \
    MODEL_CARD_URL=https://github.com/HansTeee/chest-xray-classifier/releases/download/model-v1/model_card.json \
    MODEL_SHA256=dd0bd828af28ee4a2344546bf03363d1c5d1cc4e3376cf0a9205c19c6ffa09da
WORKDIR /home/user/app

RUN mkdir -p $TORCH_HOME $MPLCONFIGDIR $STREAMLIT_HOME /home/user/app/models

# Only what serving needs (see .dockerignore). Weights are NOT baked in:
# they are downloaded from MODEL_URL (a GitHub Release asset) at startup and
# verified against MODEL_SHA256, or mounted locally into ./models.
COPY --chown=user src/__init__.py src/data.py src/model.py src/gradcam.py ./src/
COPY --chown=user api/ ./api/
COPY --chown=user ui/ ./ui/
COPY --chown=user docker/start.sh ./start.sh
COPY --chown=user docker/streamlit_config.toml /home/user/.streamlit/config.toml
RUN chmod +x ./start.sh

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=4); u.urlopen('http://localhost:7860/_stcore/health', timeout=4)" || exit 1

CMD ["./start.sh"]
