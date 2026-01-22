# syntax=docker/dockerfile:1

########################
# 1) Build stage
########################
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 AS build

ARG ENCODER=default
ENV ENCODER=${ENCODER}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev \
      build-essential git ninja-build cmake pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PIP_NO_CACHE_DIR=1
RUN python -m pip install --upgrade pip wheel setuptools

WORKDIR /app
COPY wsi-cbir/src/ /app/src/
COPY deps/ /app/deps/

## Dynamically install the correct dependencies for the selected model
RUN set -eu; \
  case "$ENCODER" in \
    CHIEF)    python -m pip install -r deps/requirements_CHIEF.txt ;; \
    TITAN)    python -m pip install -r deps/requirements_TITAN.txt ;; \
    PRISM)    python -m pip install -r deps/requirements_PRISM.txt ;; \
    GIGAPATH) python -m pip install -r deps/requirements_GIGAPATH.txt && python -m pip install flash-attn==2.8.0.post2 --no-build-isolation && python -m pip install wsidicom ;; \
    *) echo "Unknown ENCODER='$ENCODER'"; exit 1 ;; \
  esac


RUN python -m pip install fastapi uvicorn[standard] \
 && rm -rf /root/.cache/pip

########################
# 2) Runtime stage
########################
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# venv with all python deps
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app
COPY wsi-cbir/src/ /app/src/
COPY wsi-cbir/api/ /app/api/
COPY wsi-cbir/app.py /app/app.py

## Bake the CHIEF weights into the model
COPY data/model_weights/CHIEF_patch.pth /model_weights/
COPY data/model_weights/CHIEF_slide.pth /model_weights/


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "6001"]
