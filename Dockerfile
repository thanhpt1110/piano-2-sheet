# piano2sheet — CUDA 12.8 runtime image (Blackwell sm_120 compatible).
# Build:  docker build -t piano2sheet .
# Run:    docker run --rm -it --gpus all -v "$PWD:/workspace" piano2sheet \
#           python3 src/pipeline.py --audio /workspace/song.mp3 --workdir /workspace/runs/song \
#           --title "Song" --composer "Artist"
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        ffmpeg lilypond musescore3 fluidsynth fluid-soundfont-gm timidity xvfb \
        git curl ca-certificates unzip \
    && rm -rf /var/lib/apt/lists/*

# deno: JS runtime for yt-dlp's YouTube challenge solver
RUN curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh \
    && ln -sf /usr/local/bin/deno /usr/bin/deno

WORKDIR /workspace

COPY requirements.txt .
RUN python3 -m pip install --break-system-packages --upgrade pip \
    && python3 -m pip install --break-system-packages torch torchaudio \
         --index-url https://download.pytorch.org/whl/cu128 \
    && python3 -m pip install --break-system-packages -r requirements.txt \
    && python3 -m pip install --break-system-packages --no-deps transkun piano_transcription_inference

COPY . .
CMD ["bash"]
