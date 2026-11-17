# Chest X-ray pneumonia classifier: FastAPI (internal :8000) + Streamlit (:7860)
# in one CPU-only image, compatible with Hugging Face Docker Spaces
# (single exposed port 7860, runs as non-root UID 1000).
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

# HF Spaces runs the container as UID 1000. Create that user and make every
# cache/config dir it writes to live under its home.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    TORCH_HOME=/home/user/.cache/torch \
    XDG_CACHE_HOME=/home/user/.cache \
    MPLCONFIGDIR=/home/user/.cache/matplotlib \
    STREAMLIT_HOME=/home/user/.streamlit \
    API_URL=http://localhost:8000 \
    MODEL_PATH=/home/user/app/models/model.pt \
    MODEL_CARD_PATH=/home/user/app/models/model_card.json \
    TORCH_THREADS=2
WORKDIR /home/user/app

RUN mkdir -p $HF_HOME $TORCH_HOME $MPLCONFIGDIR $STREAMLIT_HOME /home/user/app/models

# Only what serving needs (see .dockerignore). Weights are NOT baked in:
# they are downloaded from HF_MODEL_REPO at startup (or mounted locally).
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
