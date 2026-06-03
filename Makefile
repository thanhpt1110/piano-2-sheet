VENV ?= .venv
PY := $(VENV)/bin/python

.PHONY: help setup fmt lint clean

help:
	@echo "make setup   - create .venv and install all dependencies (CUDA 12.8 wheels)"
	@echo "make fmt     - format src/ with black"
	@echo "make lint    - byte-compile all sources (syntax check)"
	@echo "make clean   - remove caches"

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install --no-deps transkun piano_transcription_inference

fmt:
	$(PY) -m black src

lint:
	$(PY) -m compileall -q src

clean:
	rm -rf src/__pycache__ **/__pycache__
