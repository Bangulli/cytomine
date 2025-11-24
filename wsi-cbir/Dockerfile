FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common curl ca-certificates \
      build-essential git ninja-build cmake pkg-config \
      python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    MAX_JOBS=8
RUN python -m pip install --upgrade pip wheel setuptools

WORKDIR /app
ENV PYTHONPATH=/app

# copy code
COPY wsi-cbir/src/ /app/src/
COPY wsi-cbir/api/ /app/api/
COPY wsi-cbir/app.py /app/app.py

COPY /requirements_chief.txt /app/requirements_chief.txt
RUN python -m pip install -r /app/requirements_chief.txt
RUN python -m pip install fastapi uvicorn[standard]

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "6001"] 

