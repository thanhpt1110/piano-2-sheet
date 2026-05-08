# Local Piano Audio → Sheet PDF Agent Guide

**Mục tiêu:** chạy local, không dùng cloud/subscription, lấy một link YouTube hoặc file audio piano solo, transcribe thành MIDI, rồi xuất **PDF sheet + MusicXML + MIDI**.

**Mẫu test của bạn:** `https://youtu.be/EO0g6grDd74?list=RDEO0g6grDd74`  
Khi kiểm tra link ngày 2026-06-07, YouTube hiển thị tiêu đề: **“童话 Tong hua (Fairy tale)- Guang Liang piano cover”**.

> Chỉ xử lý video/audio bạn có quyền tải xuống, chuyển đổi và lưu trữ. Không bypass DRM, paywall, login, hoặc điều khoản dịch vụ.

---

## 0. Kết luận kỹ thuật cho agent

### Chọn pipeline chính

Dùng:

```text
YouTube/audio
→ yt-dlp + ffmpeg: lấy audio WAV 44.1 kHz mono
→ Transkun V2: piano audio → MIDI
→ MuseScore Studio CLI nếu có: MIDI → MusicXML/PDF
→ fallback: music21 → MusicXML → LilyPond PDF
```

### Vì sao không “model xuất PDF trực tiếp”?

Các model accuracy cao cho piano hiện chủ yếu xuất **MIDI/event notes**, không xuất PDF đẹp trực tiếp. PDF là bước **engraving/notation rendering** riêng. Vì vậy pipeline đúng là:

```text
audio → MIDI → MusicXML → PDF
```

### Model accuracy nên ưu tiên

1. **Transkun V2**: ưu tiên chính cho solo piano audio-to-MIDI. Repo cho chạy `transkun input.mp3 output.mid --device cuda`. Model card báo trên MAESTRO V3: note onset F1 khoảng `0.9832`, onset+offset F1 khoảng `0.9349`, onset+offset+velocity F1 khoảng `0.9296`.
2. **ByteDance / qiuqiangkong HPT**: fallback/stable baseline, có pedal transcription. Repo `piano_transcription_inference` chạy CUDA và xuất MIDI.
3. **Oh Sheet**: tốt để tham khảo UI/pipeline YouTube → PDF, nhưng transcription mặc định dùng Basic Pitch, không phải lựa chọn accuracy cao nhất cho piano solo. Có thể fork và thay stage transcription bằng Transkun.

### Cấu hình GPU

- **Một bài piano đơn lẻ không cần 8 GPU**. Chạy 1 bài trên 1 GPU là đúng nhất. Multi-GPU chỉ giúp khi chạy **nhiều job song song**.
- Với **8×H200**, chạy 8 worker, mỗi worker pin vào 1 GPU.
- Với **RTX PRO 6000 Blackwell Server**, dùng làm inference server/queue rất hợp lý. Đảm bảo dùng PyTorch/CUDA mới, tránh wheel cũ không hỗ trợ Blackwell.
- Nếu chỉ test 1 link YouTube, dùng `GPU 0` là đủ; bottleneck thường là audio download/ffmpeg và PDF engraving, không phải VRAM.

---

## 1. Nguồn tham khảo đã kiểm tra

- Transkun repo: https://github.com/Yujia-Yan/Transkun
- ByteDance HPT inference repo: https://github.com/qiuqiangkong/piano_transcription_inference
- Oh Sheet repo: https://github.com/Oh-Sheet-Team/oh-sheet
- MuseScore CLI handbook: https://handbook.musescore.org/appendix/command-line-usage
- yt-dlp repo: https://github.com/yt-dlp/yt-dlp
- NVIDIA Container Toolkit: https://github.com/NVIDIA/nvidia-container-toolkit
- NVIDIA NGC PyTorch container docs: https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html
- PyTorch local install guide: https://pytorch.org/get-started/locally/
- H200 official page: https://www.nvidia.com/en-us/data-center/h200/
- RTX PRO 6000 Blackwell official page: https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/

---

## 2. Kiến trúc thư mục chuẩn

Agent tạo thư mục:

```bash
sudo mkdir -p /opt/piano2sheet/{src,runs,cache,models,logs}
sudo chown -R "$USER:$USER" /opt/piano2sheet
cd /opt/piano2sheet
```

Mỗi job tạo output riêng:

```text
/opt/piano2sheet/runs/
└── tonghua_YYYYMMDD_HHMMSS/
    ├── 00_source.url
    ├── 01_raw_audio.*
    ├── 02_audio_44100_mono.wav
    ├── 03_transkun.mid
    ├── 04_hpt.mid                      # optional
    ├── 05_score.musicxml
    ├── 06_score.pdf
    ├── run.log
    └── metadata.json
```

---

## 3. Setup khuyến nghị: Docker + NGC PyTorch

### 3.1. Host prerequisites

Trên host Linux:

```bash
nvidia-smi
docker --version
```

Nếu Docker chưa thấy GPU, cài NVIDIA Container Toolkit. NVIDIA Container Toolkit cho phép Docker container dùng GPU NVIDIA; host cần driver NVIDIA, không bắt buộc cài CUDA Toolkit trên host nếu dùng container.

Test GPU container:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

Nếu máy RTX PRO 6000 Blackwell báo lỗi CUDA kernel / architecture, dùng image NGC PyTorch mới hơn, ví dụ `nvcr.io/nvidia/pytorch:26.05-py3`, thay vì tự ghép wheel cũ.

### 3.2. Dockerfile

Tạo file:

```bash
cd /opt/piano2sheet
cat > Dockerfile <<'EOF'
FROM nvcr.io/nvidia/pytorch:26.05-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    ca-certificates \
    lilypond \
    timidity \
    fluidsynth \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# Không để pip kéo/downgrade torch trong NGC image.
RUN python -m pip install --no-deps transkun

RUN python -m pip install \
    yt-dlp \
    librosa \
    soundfile \
    mido \
    pretty_midi \
    music21 \
    numpy \
    scipy \
    tqdm \
    mir_eval \
    moduleconf \
    ffmpeg-python \
    piano_transcription_inference

WORKDIR /workspace
EOF
```

Build:

```bash
docker build -t piano2sheet:ngc .
```

Test PyTorch thấy GPU:

```bash
docker run --rm --gpus all piano2sheet:ngc python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY
```

---

## 4. Setup bare-metal thay thế

Dùng bare-metal nếu agent không được dùng Docker.

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg git curl lilypond timidity fluidsynth xvfb python3-venv python3-pip

cd /opt/piano2sheet
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Chọn command PyTorch CUDA từ trang chính thức nếu cần.
# Ví dụ CUDA 12.4:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install --no-deps transkun
pip install yt-dlp librosa soundfile mido pretty_midi music21 numpy scipy tqdm mir_eval moduleconf ffmpeg-python piano_transcription_inference
```

Verify:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
PY
```

---

## 5. Optional: cài MuseScore Studio CLI để PDF đẹp hơn

MuseScore CLI có thể convert score sang PDF bằng `-o/--export-to`. Nếu cài được MuseScore Studio trên host thì chất lượng import/export thường dễ edit hơn fallback LilyPond.

Agent check:

```bash
which musescore || true
which mscore || true
which mscore3 || true
which musescore3 || true
```

Test:

```bash
musescore --version || true
mscore --version || true
musescore3 --version || true
```

Nếu không có MuseScore, pipeline vẫn xuất MusicXML và PDF qua `music21 + musicxml2ly + lilypond`, nhưng layout có thể kém hơn và cần edit tay.

---

## 6. Script chính: `pipeline.py`

Tạo file:

```bash
cat > /opt/piano2sheet/src/pipeline.py <<'PY'
#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run(cmd, log_file, env=None, check=True):
    cmd = [str(x) for x in cmd]
    line = "+ " + " ".join(cmd)
    print(line)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        p = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed with exit={p.returncode}: {' '.join(cmd)}")
    return p.returncode


def find_any(names):
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def download_youtube(url, workdir, log_file):
    # Download best available audio/video container first; then normalize with ffmpeg.
    # Do not use --playlist for the sample link; list parameter should not pull a whole playlist.
    out_template = str(workdir / "01_raw_audio.%(ext)s")
    run(
        [
            "yt-dlp",
            "--no-playlist",
            "-f",
            "ba/b",
            "-o",
            out_template,
            url,
        ],
        log_file,
    )
    candidates = sorted(workdir.glob("01_raw_audio.*"))
    if not candidates:
        raise FileNotFoundError("yt-dlp did not create 01_raw_audio.*")
    return candidates[0]


def normalize_audio(input_path, wav_path, log_file):
    # 44.1 kHz mono matches common piano transcription assumptions.
    # Avoid aggressive denoise/time-stretch; it can damage note onsets.
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-ac",
            "1",
            "-ar",
            "44100",
            "-af",
            "aresample=async=1:first_pts=0",
            wav_path,
        ],
        log_file,
    )


def run_transkun(wav_path, midi_path, log_file, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "4")
    env["MKL_NUM_THREADS"] = env.get("MKL_NUM_THREADS", "4")
    # Inside a process where CUDA_VISIBLE_DEVICES is a single GPU, Transkun sees it as cuda.
    run(["transkun", wav_path, midi_path, "--device", "cuda"], log_file, env=env)


def run_hpt_optional(wav_path, midi_path, log_file, gpu_id):
    # Fallback model. It may be more dependency-sensitive than Transkun.
    code = f"""
import librosa
from piano_transcription_inference import PianoTranscription, sample_rate
audio, _ = librosa.load(r'{wav_path}', sr=sample_rate, mono=True)
transcriptor = PianoTranscription(device='cuda', checkpoint_path=None)
transcriptor.transcribe(audio, r'{midi_path}')
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    run([sys.executable, "-c", code], log_file, env=env)


def export_with_musescore(midi_path, musicxml_path, pdf_path, log_file):
    # Try common binary names. On Linux packages this may be musescore, mscore, musescore3, or mscore3.
    exe = find_any(["musescore", "mscore", "musescore3", "mscore3", "musescore4", "mscore4"])
    if not exe:
        return False

    # Some server installs require xvfb-run for Qt headless rendering.
    xvfb = find_any(["xvfb-run"])
    base_cmd = [exe]
    if xvfb:
        base_cmd = [xvfb, "-a", exe]

    ok = True
    try:
        run(base_cmd + [str(midi_path), "-o", str(musicxml_path)], log_file)
        run(base_cmd + [str(midi_path), "-o", str(pdf_path)], log_file)
    except Exception as e:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"MuseScore export failed: {e}\n")
        ok = False
    return ok and musicxml_path.exists() and pdf_path.exists()


def export_with_music21_lilypond(midi_path, musicxml_path, pdf_path, log_file):
    # Fallback route: MIDI -> MusicXML with music21, then MusicXML -> LilyPond -> PDF.
    # This is less pretty than MuseScore import but keeps the pipeline fully local/headless.
    code = f"""
from music21 import converter
from pathlib import Path
midi = Path(r'{midi_path}')
xml = Path(r'{musicxml_path}')
s = converter.parse(str(midi))
try:
    # Quantize best-effort. Agents should still inspect output.
    s = s.quantize()
except Exception:
    pass
s.write('musicxml', fp=str(xml))
print(xml)
"""
    run([sys.executable, "-c", code], log_file)

    ly_path = musicxml_path.with_suffix(".ly")
    run(["musicxml2ly", "-o", str(ly_path), str(musicxml_path)], log_file)

    # LilyPond uses output basename without .pdf.
    out_base = pdf_path.with_suffix("")
    run(["lilypond", "-o", str(out_base), str(ly_path)], log_file)

    if not pdf_path.exists():
        # LilyPond may choose basename from .ly; search nearby.
        candidates = list(pdf_path.parent.glob("*.pdf"))
        if candidates:
            shutil.copy2(candidates[0], pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not created: {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="YouTube/video URL")
    src.add_argument("--audio", help="Local audio file")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--gpu", default="0", help="Physical GPU id to expose to this job")
    parser.add_argument("--engine", default="transkun", choices=["transkun", "hpt", "both"])
    args = parser.parse_args()

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    log_file = workdir / "run.log"
    log_file.write_text("", encoding="utf-8")

    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "url": args.url,
        "audio": args.audio,
        "gpu": args.gpu,
        "engine": args.engine,
    }

    if args.url:
        (workdir / "00_source.url").write_text(args.url + "\n", encoding="utf-8")
        raw_audio = download_youtube(args.url, workdir, log_file)
    else:
        raw_audio = workdir / ("01_raw_audio" + Path(args.audio).suffix)
        shutil.copy2(args.audio, raw_audio)

    wav_path = workdir / "02_audio_44100_mono.wav"
    normalize_audio(raw_audio, wav_path, log_file)

    transkun_midi = workdir / "03_transkun.mid"
    hpt_midi = workdir / "04_hpt.mid"

    if args.engine in ("transkun", "both"):
        run_transkun(wav_path, transkun_midi, log_file, args.gpu)

    if args.engine in ("hpt", "both"):
        run_hpt_optional(wav_path, hpt_midi, log_file, args.gpu)

    # Default chosen output: Transkun if available; otherwise HPT.
    chosen_midi = transkun_midi if transkun_midi.exists() else hpt_midi
    if not chosen_midi.exists():
        raise FileNotFoundError("No MIDI output produced")

    musicxml_path = workdir / "05_score.musicxml"
    pdf_path = workdir / "06_score.pdf"

    if not export_with_musescore(chosen_midi, musicxml_path, pdf_path, log_file):
        export_with_music21_lilypond(chosen_midi, musicxml_path, pdf_path, log_file)

    metadata.update(
        {
            "raw_audio": str(raw_audio),
            "wav": str(wav_path),
            "chosen_midi": str(chosen_midi),
            "musicxml": str(musicxml_path),
            "pdf": str(pdf_path),
        }
    )
    (workdir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("DONE")
    print("MIDI:", chosen_midi)
    print("MusicXML:", musicxml_path)
    print("PDF:", pdf_path)


if __name__ == "__main__":
    main()
PY

chmod +x /opt/piano2sheet/src/pipeline.py
```

---

## 7. Chạy mẫu với link YouTube của bạn

### 7.1. Docker, 1 GPU

```bash
cd /opt/piano2sheet

JOB="tonghua_$(date +%Y%m%d_%H%M%S)"
mkdir -p "runs/$JOB"

docker run --rm \
  --gpus '"device=0"' \
  --ipc=host \
  -v /opt/piano2sheet:/workspace \
  -w /workspace \
  piano2sheet:ngc \
  python /workspace/src/pipeline.py \
    --url "https://youtu.be/EO0g6grDd74?list=RDEO0g6grDd74" \
    --workdir "/workspace/runs/$JOB" \
    --gpu 0 \
    --engine transkun
```

Output:

```bash
ls -lah "/opt/piano2sheet/runs/$JOB"
```

Bạn cần thấy:

```text
03_transkun.mid
05_score.musicxml
06_score.pdf
run.log
metadata.json
```

### 7.2. Bare-metal

```bash
cd /opt/piano2sheet
source .venv/bin/activate

JOB="tonghua_$(date +%Y%m%d_%H%M%S)"
mkdir -p "runs/$JOB"

CUDA_VISIBLE_DEVICES=0 python src/pipeline.py \
  --url "https://youtu.be/EO0g6grDd74?list=RDEO0g6grDd74" \
  --workdir "runs/$JOB" \
  --gpu 0 \
  --engine transkun
```

---

## 8. A/B test Transkun vs HPT cho cùng bài

Chạy cả hai:

```bash
cd /opt/piano2sheet

JOB="tonghua_ab_$(date +%Y%m%d_%H%M%S)"
mkdir -p "runs/$JOB"

docker run --rm \
  --gpus '"device=0"' \
  --ipc=host \
  -v /opt/piano2sheet:/workspace \
  -w /workspace \
  piano2sheet:ngc \
  python /workspace/src/pipeline.py \
    --url "https://youtu.be/EO0g6grDd74?list=RDEO0g6grDd74" \
    --workdir "/workspace/runs/$JOB" \
    --gpu 0 \
    --engine both
```

Sau đó mở:

```text
03_transkun.mid
04_hpt.mid
05_score.musicxml
06_score.pdf
```

Nếu `04_hpt.mid` hay hơn đoạn nào, agent có thể export PDF lại từ HPT bằng cách đổi `chosen_midi` trong script hoặc chạy thủ công phần export.

---

## 9. Tối ưu cho 8×H200

### 9.1. Nguyên tắc

Không dùng `torchrun`, DDP, tensor parallel cho một bài piano. Transkun/HPT inference là job nhỏ so với 8×H200. Cách nhanh nhất để tận dụng 8 GPU là **song song theo bài**, không phải chia 1 bài ra 8 GPU.

### 9.2. Cấu hình host

Khuyến nghị:

```bash
sudo nvidia-smi -pm 1
nvidia-smi topo -m
```

Không bắt buộc bật MIG. Với workload này, MIG chỉ hữu ích nếu cluster scheduler cần chia GPU cho nhiều tenant; nếu máy là của bạn, để full GPU đơn giản hơn.

### 9.3. Chạy batch URL với 8 worker

Tạo `urls.txt`:

```text
https://youtu.be/EO0g6grDd74?list=RDEO0g6grDd74
```

Tạo batch runner:

```bash
cat > /opt/piano2sheet/src/run_batch_8gpu.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

URL_FILE="${1:-urls.txt}"
ROOT="/opt/piano2sheet"
IMAGE="piano2sheet:ngc"

mkdir -p "$ROOT/runs"

idx=0
while IFS= read -r url; do
  if [ -z "$url" ]; then
    continue
  fi

  gpu=$((idx % 8))
  job="job_${idx}_gpu${gpu}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$ROOT/runs/$job"

  echo "Launching idx=$idx gpu=$gpu url=$url job=$job"

  docker run --rm \
    --gpus "\"device=${gpu}\"" \
    --ipc=host \
    -v "$ROOT:/workspace" \
    -w /workspace \
    "$IMAGE" \
    python /workspace/src/pipeline.py \
      --url "$url" \
      --workdir "/workspace/runs/$job" \
      --gpu 0 \
      --engine transkun \
      > "$ROOT/runs/$job/console.log" 2>&1 &

  idx=$((idx + 1))

  # Keep at most 8 concurrent jobs.
  while [ "$(jobs -r | wc -l)" -ge 8 ]; do
    sleep 2
  done
done < "$URL_FILE"

wait
echo "All jobs finished."
SH

chmod +x /opt/piano2sheet/src/run_batch_8gpu.sh
```

Chạy:

```bash
cd /opt/piano2sheet
/opt/piano2sheet/src/run_batch_8gpu.sh urls.txt
```

Lưu ý: trong Docker command, `--gpus "device=${gpu}"` chỉ expose một physical GPU vào container. Bên trong container, GPU đó là logical `cuda:0`, nên script dùng `--gpu 0`.

### 9.4. Khi nào nên dùng H200?

Dùng H200 nếu:

- cần chạy hàng trăm/hàng nghìn bài piano song song;
- cần queue nội bộ cho nhiều người;
- cần chạy thêm model LLM/agent orchestration bên cạnh transcription;
- muốn benchmark/fine-tune AMT model.

Với 1 bài 3–5 phút, H200 không làm PDF “đẹp hơn”. Chất lượng phụ thuộc model + postprocess + engraving.

---

## 10. Tối ưu cho RTX PRO 6000 Blackwell Server

RTX PRO 6000 Blackwell có 96 GB GDDR7 ECC theo thông số NVIDIA. Nó đủ dư cho Transkun/HPT. Lợi thế chính là làm server inference luôn bật, không lãng phí H200.

### 10.1. Khuyến nghị

- Dùng Docker NGC PyTorch mới, ví dụ `nvcr.io/nvidia/pytorch:26.05-py3`.
- Tránh môi trường PyTorch quá cũ. Nếu gặp lỗi kiểu `no kernel image is available for execution on the device`, gần như chắc là wheel/container chưa hỗ trợ kiến trúc Blackwell.
- Không để `pip install transkun` downgrade `torch`. Dùng `pip install --no-deps transkun` trong image đã có PyTorch mới.
- Với 2 máy, nên chia vai:
  - **RTX PRO 6000 Blackwell Server**: service/API/queue online.
  - **8×H200**: batch throughput/fine-tuning/benchmark lớn.

### 10.2. Chạy server queue đơn giản

Agent có thể triển khai một queue bằng filesystem:

```text
/opt/piano2sheet/queue/incoming/*.url
/opt/piano2sheet/queue/running/
/opt/piano2sheet/queue/done/
/opt/piano2sheet/queue/failed/
```

Worker loop:

```bash
cat > /opt/piano2sheet/src/queue_worker.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

ROOT="/opt/piano2sheet"
GPU_ID="${GPU_ID:-0}"
IMAGE="piano2sheet:ngc"

mkdir -p "$ROOT/queue/incoming" "$ROOT/queue/running" "$ROOT/queue/done" "$ROOT/queue/failed" "$ROOT/runs"

while true; do
  file="$(find "$ROOT/queue/incoming" -maxdepth 1 -type f -name '*.url' | head -n 1 || true)"
  if [ -z "$file" ]; then
    sleep 5
    continue
  fi

  base="$(basename "$file" .url)"
  running="$ROOT/queue/running/$base.url"
  mv "$file" "$running"

  url="$(cat "$running")"
  job="${base}_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$ROOT/runs/$job"

  set +e
  docker run --rm \
    --gpus "\"device=${GPU_ID}\"" \
    --ipc=host \
    -v "$ROOT:/workspace" \
    -w /workspace \
    "$IMAGE" \
    python /workspace/src/pipeline.py \
      --url "$url" \
      --workdir "/workspace/runs/$job" \
      --gpu 0 \
      --engine transkun \
      > "$ROOT/runs/$job/console.log" 2>&1
  code=$?
  set -e

  if [ "$code" -eq 0 ]; then
    mv "$running" "$ROOT/queue/done/$base.url"
  else
    mv "$running" "$ROOT/queue/failed/$base.url"
  fi
done
SH

chmod +x /opt/piano2sheet/src/queue_worker.sh
```

Run one worker per GPU:

```bash
GPU_ID=0 nohup /opt/piano2sheet/src/queue_worker.sh > /opt/piano2sheet/logs/worker0.log 2>&1 &
GPU_ID=1 nohup /opt/piano2sheet/src/queue_worker.sh > /opt/piano2sheet/logs/worker1.log 2>&1 &
```

Submit sample:

```bash
echo "https://youtu.be/EO0g6grDd74?list=RDEO0g6grDd74" > /opt/piano2sheet/queue/incoming/tonghua.url
```

---

## 11. Option Oh Sheet: dùng UI/pipeline, nhưng thay model

Oh Sheet là repo open-source có luồng:

```text
YouTube URL / MP3 / MIDI
→ ingest
→ transcribe
→ arrange
→ humanize
→ engrave
→ PDF + MusicXML + MIDI
```

Repo nói hỗ trợ YouTube URL, Basic Pitch transcription, optional Demucs, music21 → MusicXML + LilyPond → PDF.

Nếu muốn UI gần Songscription:

1. Clone Oh Sheet.
2. Chạy theo README để có frontend/backend.
3. Không dùng Basic Pitch mặc định cho piano solo nếu mục tiêu accuracy cao nhất.
4. Patch stage `backend/services/transcribe.py` để gọi `transkun` và trả về MIDI.

Pseudo patch trong service:

```python
import subprocess
from pathlib import Path

def transcribe_with_transkun(audio_path: Path, out_midi: Path, gpu: int = 0) -> Path:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    subprocess.check_call(
        ["transkun", str(audio_path), str(out_midi), "--device", "cuda"],
        env=env,
    )
    return out_midi
```

Cách này tận dụng Oh Sheet cho ingest/UI/artifact, nhưng vẫn lấy transcription tốt hơn cho solo piano.

---

## 12. Post-processing để sheet “giống Songscription” hơn

Model chỉ tạo MIDI. Sheet đẹp cần các bước sau:

### 12.1. Quantize

- Nếu nhạc ballad piano rubato, quantize quá mạnh sẽ làm sai feel.
- Bản đầu tiên nên export raw MIDI, rồi import vào MuseScore với quantization `1/16` hoặc `1/32`.
- Nếu sheet quá rối, thử tăng quantization lên `1/8` cho bản beginner.

### 12.2. Split tay trái/tay phải

MIDI từ Transkun thường là một track piano với pitch/time/velocity. MuseScore import có thể tự tách staff piano, nhưng không phải lúc nào đúng.

Rule cơ bản nếu tự xử lý:

```text
pitch < C4  → left hand
pitch >= C4 → right hand
```

Rule tốt hơn:

- bass note thấp nhất mỗi beat → left hand;
- melody/top line → right hand;
- notes giữa chia theo khoảng cách tay và voice continuity.

### 12.3. Key signature/time signature

AMT model thường không biết key/time signature chính xác từ audio. Agent nên:

1. dùng `music21` analyze key;
2. dùng beat tracking nếu cần;
3. vẫn cho người dùng edit trong MuseScore/Dorico nếu output final quan trọng.

### 12.4. Pedal

- Transkun checkpoint mặc định trong pip là “No Pedal Ext” theo README. Nó có lợi cho performance thực tế nhưng sustain/pedal notation có thể không đầy đủ.
- HPT có pedal transcription tốt hơn nếu bạn cần pedal marks.
- Cho bản PDF dễ đọc, pedal có thể để mặc định hoặc thêm thủ công.

### 12.5. Human review bắt buộc

Không có open-source local pipeline nào đảm bảo PDF hoàn hảo như người chép nhạc. Agent nên luôn output cả:

```text
raw MIDI
MusicXML
PDF
```

để người dùng sửa MusicXML trong MuseScore/Dorico/Sibelius.

---

## 13. Benchmark nhanh cho agent

Sau khi chạy sample, agent kiểm tra:

```bash
JOB_DIR="/opt/piano2sheet/runs/<JOB_NAME>"

test -s "$JOB_DIR/03_transkun.mid"
test -s "$JOB_DIR/05_score.musicxml"
test -s "$JOB_DIR/06_score.pdf"

ls -lh "$JOB_DIR"
tail -n 80 "$JOB_DIR/run.log"
```

Kiểm tra MIDI có nhiều note hợp lý:

```bash
python - <<'PY'
import pretty_midi, sys
pm = pretty_midi.PrettyMIDI(sys.argv[1])
notes = sum(len(i.notes) for i in pm.instruments)
duration = pm.get_end_time()
print("notes:", notes)
print("duration_sec:", duration)
for inst in pm.instruments:
    print(inst.name, inst.program, len(inst.notes))
PY /opt/piano2sheet/runs/<JOB_NAME>/03_transkun.mid
```

Nếu note count quá ít hoặc duration sai:

- kiểm tra `02_audio_44100_mono.wav`;
- mở `run.log`;
- update `yt-dlp`;
- thử input local WAV thay vì YouTube;
- thử HPT.

---

## 14. Troubleshooting

### 14.1. `torch.cuda.is_available() == False`

Check:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

Nếu host OK nhưng container fail, cài/cấu hình NVIDIA Container Toolkit.

### 14.2. Blackwell lỗi kernel/CUDA

Dấu hiệu:

```text
no kernel image is available for execution on the device
```

Fix:

- dùng NGC PyTorch container mới;
- không dùng torch wheel cũ;
- không để pip package downgrade torch;
- verify bằng `torch.cuda.get_device_name(0)`.

### 14.3. yt-dlp fail với YouTube

Fix:

```bash
python -m pip install -U --pre "yt-dlp[default]"
yt-dlp --version
```

Nếu video cần login/cookie, chỉ dùng cookie khi bạn có quyền truy cập và tuân thủ điều khoản.

### 14.4. MuseScore không chạy trong server/headless

Fix:

```bash
xvfb-run -a musescore input.mid -o output.pdf
```

Nếu không có MuseScore, dùng fallback LilyPond trong script.

### 14.5. PDF quá rối

Không phải lỗi GPU. Là lỗi notation/postprocess. Hành động:

1. thử import MIDI vào MuseScore thủ công;
2. quantize 1/16 hoặc 1/8;
3. tách tay trái/phải;
4. giảm grace notes/very short notes;
5. sửa meter/key signature.

### 14.6. Audio không phải solo piano

Nếu audio có vocal/drums/bass, Transkun sẽ sai. Với piano cover solo như link mẫu thì không cần Demucs. Nếu có mix nhiều nhạc cụ, chạy Demucs trước nhưng chất lượng sheet piano sẽ phụ thuộc stem separation.

---

## 15. Systemd service cho server queue

Tạo service cho RTX PRO 6000 Server hoặc H200 node:

```bash
sudo tee /etc/systemd/system/piano2sheet-worker0.service >/dev/null <<'EOF'
[Unit]
Description=Piano2Sheet Worker GPU0
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=simple
Environment=GPU_ID=0
WorkingDirectory=/opt/piano2sheet
ExecStart=/opt/piano2sheet/src/queue_worker.sh
Restart=always
RestartSec=5
User=YOUR_USER
Group=YOUR_USER

[Install]
WantedBy=multi-user.target
EOF
```

Sửa `YOUR_USER`, rồi:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now piano2sheet-worker0
sudo systemctl status piano2sheet-worker0
```

Cho 8×H200, tạo `worker0` đến `worker7`, mỗi service set `GPU_ID=0..7`.

---

## 16. Checklist cuối cho agent

Agent hoàn tất khi có:

```text
[ ] Docker image hoặc venv chạy được
[ ] torch.cuda.is_available() == True
[ ] yt-dlp tải được audio mẫu
[ ] ffmpeg tạo 02_audio_44100_mono.wav
[ ] Transkun tạo 03_transkun.mid
[ ] Export tạo 05_score.musicxml
[ ] Export tạo 06_score.pdf
[ ] run.log không có traceback
[ ] metadata.json ghi đầy đủ source URL và artifact paths
```

Final response cho user nên trả:

```text
Output directory: /opt/piano2sheet/runs/<JOB_NAME>
MIDI: 03_transkun.mid
MusicXML: 05_score.musicxml
PDF: 06_score.pdf
Notes: PDF cần review/cleanup nếu rhythm/hand split chưa đẹp.
```

---

## 17. Khuyến nghị vận hành thực tế

- Với bài mẫu `Tong hua`, chạy **Transkun trước**.
- Không dùng Demucs nếu input là solo piano.
- Không dùng NotaGen cho bài toán này; NotaGen là symbolic music generation, không phải audio-to-sheet transcription.
- Để output đẹp nhất, dùng **Transkun MIDI + MuseScore/Dorico manual cleanup**.
- Nếu muốn sản phẩm self-host giống Songscription, fork Oh Sheet làm UI và thay Basic Pitch bằng Transkun.
- 8×H200 là overkill cho single-song inference; dùng nó cho batch hoặc fine-tuning. RTX PRO 6000 Blackwell Server đủ làm service production cho nhu cầu này.
